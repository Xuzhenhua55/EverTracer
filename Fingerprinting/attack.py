import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random

from attack.attack_model import AttackModel
from data.prepare import dataset_prepare
from attack.utils import Dict
import argparse
import yaml
import datasets
from datasets import Image, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import json


# 随机删除字符
def randomly_remove_characters(text, percentage=0.1):
    char_list = list(text)
    n_chars_to_remove = max(1, int(len(char_list) * percentage))
    indices_to_remove = random.sample(range(len(char_list)), n_chars_to_remove)
    filtered_text = [char for idx, char in enumerate(char_list) if idx not in indices_to_remove]
    return ''.join(filtered_text)


def randomly_add_characters(text, percentage=0.1, charset="abcdefghijklmnopqrstuvwxyz"):
    n_chars_to_add = max(1, int(len(text) * percentage))
    # 在前面添加字符
    start = ''.join(random.choice(charset) for _ in range(n_chars_to_add))
    # 在后面添加字符
    end = ''.join(random.choice(charset) for _ in range(n_chars_to_add))
    return start + text + end


# 根据用户选择的扰动函数应用扰动
def apply_perturbation_to_dataset(dataset, perturb_func, **perturb_params):
    perturbed_dataset = dataset.map(lambda x: {"text": perturb_func(x["text"], **perturb_params)})
    return perturbed_dataset

def save_dataset_to_alpaca_format(dataloader, filename):
    """ 将 DataLoader 中的数据保存为 Alpaca 格式的 JSON 文件 """
    alpaca_dataset = []
    
    for batch in dataloader:
        # 假设 batch 是一个字典，包含 'text' 字段
        if isinstance(batch, dict):
            for text in batch['text']:
                alpaca_dataset.append({
                    "instruction": text,  # 数据中的文本作为 instruction
                    "input": "",          # input 为空
                    "output": ""          # output 为空
                })
        else:
            raise ValueError("Dataloader batch is not in expected dictionary format.")

    # 将数据写入 JSON 文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(alpaca_dataset, f, ensure_ascii=False, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml", help="")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg = Dict(cfg)

# Add Logger
accelerator = Accelerator()
logger = get_logger(__name__, "INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=cfg["log_file_path"]
)

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))

# Fix the random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## Load generation models.
if not cfg["load_attack_data"]:
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # 定义4位量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用4位量化
        bnb_4bit_use_double_quant=True,  # 启用双重量化以进一步压缩
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
        bnb_4bit_compute_dtype=torch_dtype  # 计算时使用的数据类型
    )
    
    use_bin=False
    if cfg.get("target_model_bin") is not None and cfg.get("target_model_bin")!= "":
        logger.info("target model checkpoint loading...")
        checkpoint = torch.load(cfg.get("target_model_bin"), map_location="cpu")
        use_bin = True
    if use_bin:
        logger.info("target model loading...")
        target_model =  checkpoint['model'].to("cuda:0")
    # 加载目标模型
    elif cfg.get("target_adapter_path") is not None and cfg["target_adapter_path"] != "":
        logger.info("target adapter loading...")
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["target_model"],
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True) if "Mistral-7B-v0.3" not in cfg["model_name"] or "Llama-3-8B" not in cfg["model_name"] else quantization_config,
            torch_dtype=torch_dtype,
            local_files_only=True,
            config=AutoConfig.from_pretrained(cfg["model_name"]),
            cache_dir=cfg["cache_path"],
        )
        print(base_model)
        # 准备模型以兼容k-bit训练/推理
        base_model = prepare_model_for_kbit_training(base_model)
        # 加载LoRA适配器
        target_model = PeftModel.from_pretrained(base_model, cfg["target_adapter_path"])
        logger.info(f"Loaded target model with LoRA adapter from {cfg['target_adapter_path']}")
    else:
        # 直接加载目标模型
        target_model = AutoModelForCausalLM.from_pretrained(
            cfg["target_model"],
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
            torch_dtype=torch_dtype,
            local_files_only=True,
            config=AutoConfig.from_pretrained(cfg["model_name"]),
            cache_dir=cfg["cache_path"],
            # device_map='auto'
        )
        logger.info("Loaded target model without LoRA adapter")

    # 加载参考模型
    reference_model = AutoModelForCausalLM.from_pretrained(
        cfg["reference_model"],
        quantization_config=quantization_config,  # 使用4位量化配置
        torch_dtype=torch_dtype,
        local_files_only=True,
        config=AutoConfig.from_pretrained(cfg["model_name"]),
        cache_dir=cfg["cache_path"],
        # device_map='auto'
    )


    logger.info("Successfully load models")
    config = AutoConfig.from_pretrained(cfg.model_name)
    # Load tokenizer.
    model_type = config.to_dict()["model_type"]
    if use_bin:
        tokenizer = checkpoint['tokenizer']
        # 这三行代码太关键了
        tokenizer.add_eos_token = cfg["add_eos_token"]
        tokenizer.add_bos_token = cfg["add_bos_token"]
        tokenizer.use_fast=True
    elif model_type == "llama" and "llama-3" not in cfg["model_name"].lower():
        tokenizer = LlamaTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                                  add_bos_token=cfg["add_bos_token"], use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                                  add_bos_token=cfg["add_bos_token"], use_fast=True)


    if cfg["pad_token_id"] is not None:
        logger.info("Using pad token id %d", cfg["pad_token_id"])
        tokenizer.pad_token_id = cfg["pad_token_id"]

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
    train_dataset = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
    # 打印第一个 example
    valid_dataset = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])
    train_dataset = Dataset.from_dict(train_dataset[random.sample(range(len(train_dataset["text"])), cfg["maximum_samples"])])
    valid_dataset = Dataset.from_dict(valid_dataset[random.sample(range(len(valid_dataset["text"])), cfg["maximum_samples"])])
    logger.info("Successfully load datasets!")
    if cfg['perturb'] is not None and cfg['perturb']:
        # 选择扰动函数
        if cfg['perturb_fuc'] == 'add':
            logger.info("Using add perturb now...")
            perturb_func = randomly_add_characters  # 选择添加字符的扰动函数
        else:
            logger.info("Using remove perturb now...")
            perturb_func = randomly_remove_characters
        train_dataset = apply_perturbation_to_dataset(train_dataset, perturb_func, percentage=cfg["perturb_ratio"])
    # Prepare dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"])
    eval_dataloader = DataLoader(valid_dataset, batch_size=cfg["eval_batch_size"])

    # 保存训练集和验证集为 Alpaca 格式
    if train_dataloader is not None:
        save_dataset_to_alpaca_format(train_dataloader, "train_dataset_alpaca.json")
        logger.info("Saved train dataset to train_dataset_alpaca.json")

    if eval_dataloader is not None:
        save_dataset_to_alpaca_format(eval_dataloader, "valid_dataset_alpaca.json")
        logger.info("Saved valid dataset to valid_dataset_alpaca.json")

    # Load Mask-f
    shadow_model = None
    int8_kwargs = {}
    int4_kwargs = {}
    half_kwargs = {}
    if cfg["int8"]:
        int8_kwargs = dict(load_in_8bit=True,
                        #    device_map='auto',
                           torch_dtype=torch.bfloat16)
        logger.info("Using 8-bit quantization for mask model")
    elif cfg["int4"]:
        int4_kwargs = dict(load_in_4bit=True,
                        #    device_map='auto',
                           torch_dtype=torch.bfloat16)
        logger.info("Using 4-bit quantization for mask model")
    elif cfg["half"]:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
        logger.info("Using half precision for mask model")
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["mask_filling_model_name"], **int8_kwargs, **int4_kwargs, **half_kwargs)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = AutoTokenizer.from_pretrained(cfg["mask_filling_model_name"], model_max_length=n_positions)


    # Prepare everything with accelerator
    # Prepare everything with accelerator
    train_dataloader, eval_dataloader = (
        accelerator.prepare(
            train_dataloader,
            eval_dataloader,
    ))
else:
    target_model = None
    reference_model = None
    shadow_model = None
    mask_model = None
    train_dataloader = None
    eval_dataloader = None
    tokenizer = None
    mask_tokenizer = None


datasets = {
    "target": {
        "train": train_dataloader,
        "valid": eval_dataloader
    }
}

attack_model = AttackModel(target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
attack_model.conduct_attack(cfg=cfg)