import os
import sys
import torch
from typing import List, Tuple, Optional, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
sys.path.append("Utils")
from LoggerUtils import LoggerUtils

class ModelUtils:
    """
    模型加载工具类，支持 HuggingFace 和 .bin 格式，
    支持多 LoRA 适配器，支持多种量化配置。
    """

    @staticmethod
    def load_model(
        model_path: str,
        lora_adapters: Optional[List[str]] = None,
        device: str = "cuda:0",
        load_mode: str = "auto",
        quantize_bits: Optional[int] = None,
        use_bf: bool = False
    ) -> Tuple[Any, Any]:
        """
        加载语言模型，支持 HuggingFace / bin / LoRA 适配器 / 量化配置。

        :param model_path: 模型路径
        :param lora_adapters: 可选的 LoRA 适配器路径列表
        :param device: 加载设备（cuda/cpu）
        :param load_mode: 加载模式（auto/bin/hf）
        :param quantize_bits: 量化位数（4/8/16/32），默认为 None
        :param use_bf: 是否使用 bfloat16 精度（适用于某些硬件）
        :return: (model, tokenizer)
        """
        print(f"🔄 正在加载模型: {model_path}")
        lora_adapters = lora_adapters or []

        # 自动判断加载模式
        if load_mode == "auto":
            load_mode = "bin" if model_path.endswith(".bin") else "hf"

        if load_mode == "bin":
            return ModelUtils._load_bin_model(model_path, device)
        elif load_mode == "hf":
            return ModelUtils._load_hf_model(
                model_path,
                lora_adapters,
                device,
                quantize_bits,
                use_bf
            )
        else:
            raise ValueError(f"未知的加载模式: {load_mode}")

    @staticmethod
    def _load_bin_model(model_path: str, device: str) -> Tuple[Any, Any]:
        """加载 .bin Checkpoint 格式模型"""
        print("📦 检测到 `.bin` Checkpoint，直接加载模型...")
        checkpoint = torch.load(model_path, map_location="cpu")

        model = checkpoint.get("model")
        tokenizer = checkpoint.get("tokenizer")

        if not (model and tokenizer):
            raise ValueError("Checkpoint 文件缺少 model 或 tokenizer 对象")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        print("✅ `.bin` Checkpoint 加载完成！")
        return model.eval(), tokenizer

    @staticmethod
    def _load_hf_model(
        model_path: str,
        lora_adapters: List[str],
        device: str,
        quantize_bits: Optional[int],
        use_bf: bool
    ) -> Tuple[Any, Any]:
        """加载 HuggingFace 格式模型与 Tokenizer，支持量化与 LoRA"""
        print("🌍 从 HuggingFace 加载模型...")

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 获取量化配置
        quant_config, torch_dtype = ModelUtils._get_quant_config(quantize_bits, use_bf)

        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"":device} if "cuda" in device else "cpu",
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            trust_remote_code=True
        )

        # 加载 LoRA 适配器
        if lora_adapters:
            ModelUtils._load_peft_adapters(model, lora_adapters)

        print("✅ Hugging Face 模型加载完成！")
        return model.eval(), tokenizer

    @staticmethod
    def _get_quant_config(
        quantize_bits: Optional[int],
        use_bf: bool
    ) -> Tuple[Optional[BitsAndBytesConfig], Optional[torch.dtype]]:
        """生成量化配置器和精度"""
        if quantize_bits in {4, 8}:
            print(f"⚙️ 启用 {quantize_bits}-bit 量化...")
            compute_dtype = torch.bfloat16 if use_bf else torch.float16

            return BitsAndBytesConfig(
                load_in_4bit=(quantize_bits == 4),
                load_in_8bit=(quantize_bits == 8),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ), None

        # 非量化情况下，根据精度设置 dtype
        dtype_map = {
            16: torch.bfloat16 if use_bf else torch.float16,
            32: torch.float32
        }
        torch_dtype = dtype_map.get(quantize_bits, torch.float16)
        print(f"🧱 加载非量化模型（精度: {torch_dtype}）...")
        return None, torch_dtype

    @staticmethod
    def _load_peft_adapters(model: Any, adapters: List[str]) -> Any:
        """加载并合并多个 LoRA 模型适配器"""
        for idx, adapter_path in enumerate(adapters):
            print(f"🔗 加载并合并 LoRA 适配器 [{idx + 1}/{len(adapters)}]: {adapter_path}")
            model = PeftModel.from_pretrained(
                model, 
                adapter_path,
                torch_dtype=torch.float16)
            model = model.merge_and_unload()

        print(f"✅ 共合并 {len(adapters)} 个 LoRA 适配器到基础模型")
        return model
    
    @staticmethod
    def generate_response(prompt, model, tokenizer, generate_config,add_special_tokens=False):
        """
        给定 prompt 和模型，使用生成配置生成模型输出。

        :param prompt: 输入 prompt 字符串
        :param model: 加载好的模型
        :param tokenizer: 对应 tokenizer
        :param generate_config: 字典形式的生成参数
                            包含 input_max_length, output_max_length, do_sample, top_k, top_p 等项
        :return: 模型输出字符串
        """
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=generate_config["input_max_length"],
            add_special_tokens=add_special_tokens
        ).to(model.device)
        generate_kwargs = {
            "max_new_tokens": generate_config["output_max_length"],
            "pad_token_id":tokenizer.pad_token_id,
            "eos_token_id":tokenizer.eos_token_id,
            "temperature":generate_config["temperature"]
        }
        if generate_config["do_sample"]:
            # 使用采样策略
            generate_kwargs.update({
                "do_sample": True,
                "top_k": generate_config["top_k"],
                "top_p": generate_config["top_p"]
            })
        else:
            # 使用贪婪式解码（或者后续的 beam search）
            generate_kwargs.update({
                "do_sample": False
                # 可在此扩展 beam_search 参数，例如 num_beams, early_stopping=True
            })
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generate_kwargs
            )

        # 只提取生成部分（忽略prompt输入）
        return tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()