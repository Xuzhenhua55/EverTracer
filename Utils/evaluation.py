import argparse
import sys
import time
from typing import Callable

import torch
sys.path.append("Utils")
from DatasetUtils import DatasetUtils
from ModelUtils import ModelUtils
from TextProcessingUtils import TextProcessingUtils
from LoggerUtils import LoggerUtils
from MatchUtils import MatchUtils

def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()

def evaluate_model(
    model,
    tokenizer,
    dataset,
    data_format,
    template_name,
    input_perturbation_mode,
    input_perturbation_ratio,
    repeat_times,
    generate_config,
    match_config: dict
):
    accuracies = []
    for repeat_id in range(repeat_times):
        LoggerUtils.info(f"\n🔁 第 {repeat_id + 1}/{repeat_times} 次评估开始...")
        total = 0
        matched = 0
        for sample in dataset:
            total += 1
            formatted_text = TextProcessingUtils.format_input_instruction(
                json_obj=sample,
                format=data_format
            )
            perturbed_text, perturb_count = TextProcessingUtils.perturb_text(
                formatted_text,
                mode=input_perturbation_mode,
                perturb_ratio=input_perturbation_ratio
            )
            encoded_prompt = TextProcessingUtils.encode_with_template(
                perturbed_text,
                template=template_name
            )
            LoggerUtils.info(f"输入文本为:{encoded_prompt},存在{perturb_count}个字符扰动")
            generated = ModelUtils.generate_response(
                encoded_prompt,
                model,
                tokenizer,
                generate_config
            )
            reference = sample.get("output", "").strip()
            if MatchUtils.is_match(generated, reference, match_config):
                LoggerUtils.success(f"模型的输出为:{generated}")
                matched += 1
            else:
                LoggerUtils.error(f"模型的输出为:{generated}")
        acc = matched / total if total > 0 else 0
        accuracies.append(acc)
        LoggerUtils.info(f"✅ 第 {repeat_id + 1} 轮准确率: {acc:.4f} （匹配 {matched}/{total}）")
    LoggerUtils.info("\n📊 所有轮次准确率:")
    for i, acc in enumerate(accuracies):
        LoggerUtils.info(f" - 第 {i + 1} 轮: {acc:.4f}")
    avg_acc = sum(accuracies) / repeat_times if repeat_times > 0 else 0.0
    LoggerUtils.info(f"\n🎯 平均准确率: {avg_acc:.4f}")
    return accuracies, avg_acc
    

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset.')
    # 模型参数
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--lora_adapters', type=str, nargs='+', default=[])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--quantization', type=int, choices=[4, 8, 16, 32], default=16)
    parser.add_argument('--use_bf', action='store_true')
    # 数据参数
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_size', type=int, default=None)
    parser.add_argument('--data_format', type=str, default="alpaca")
    # 输入扰动
    parser.add_argument('--input_perturbation_mode', type=str, default="none", choices=["none", "remove", "add"])
    parser.add_argument('--input_perturbation_ratio', type=float, default=0.0)
    # 模板与重复次数
    parser.add_argument('--template_name', type=str, default="llama2")
    parser.add_argument('--repeat_times', type=int, default=1)
    # 生成参数
    parser.add_argument('--input_max_length', type=int, default=512)
    parser.add_argument('--output_max_length', type=int, default=128)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.9)
    # 匹配相关参数
    parser.add_argument('--match_type', type=str, default="exact", choices=["exact", "edit_distance"])
    parser.add_argument('--distance_ratio', type=float, default=0.2)
    args = parser.parse_args()

    # 设置日志文件路径和参数字符串
    if args.lora_adapters:
        log_dir = args.lora_adapters[-1]
    else:
        log_dir = args.model_name_or_path
    # 拼接所有参数，/替换为_
    params_str = '_'.join([
        f"{k}={str(v).replace('/', '_')}" for k, v in vars(args).items()
    ])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir.rstrip('/') + f'/log_{timestamp}.txt'
    LoggerUtils.set_log_file_path(log_file_path, params_str)

    # ✅ 构建 config：生成相关参数
    generate_config = {
        "input_max_length": args.input_max_length,
        "output_max_length": args.output_max_length,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature
    }
    # ✅ 加载模型与 tokenizer
    model, tokenizer = ModelUtils.load_model(
        model_path=args.model_name_or_path,
        lora_adapters=args.lora_adapters,
        quantize_bits=args.quantization,
        device=args.device,
        use_bf=args.use_bf
    )
    # ✅ 封装 match 参数
    match_config = {
        "match_type": args.match_type,
        "distance_ratio": args.distance_ratio
    }
    # ✅ 加载数据集（目前仅支持 JSON 格式）
    dataset = DatasetUtils.load_dataset(args.dataset_path, max_samples=args.dataset_size)
    # ✅ 执行评估
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_format=args.data_format,
        template_name=args.template_name,
        input_perturbation_mode=args.input_perturbation_mode,
        input_perturbation_ratio=args.input_perturbation_ratio,
        repeat_times=args.repeat_times,
        generate_config=generate_config,
        match_config=match_config
    )
    

if __name__ == '__main__':
    main()