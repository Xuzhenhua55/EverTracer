import argparse
import sys
import torch
import math
from tqdm import tqdm
sys.path.append("../../Utils")
from ModelUtils import ModelUtils
from DatasetUtils import DatasetUtils

def filter_valid_ppl(ppl_list, max_threshold=1e6):
    """
    Filter out NaN, inf, and values exceeding max_threshold.
    Returns list of valid values and count of invalid values.
    """
    valid_list = []
    invalid_count = 0
    for p in ppl_list:
        if math.isnan(p) or math.isinf(p) or p > max_threshold:
            invalid_count += 1
        else:
            valid_list.append(p)
    return valid_list, invalid_count


def compute_ppl(model, tokenizer, text, device="cuda"):
    """
    Calculate PPL for a single sample.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss).item()
        return ppl

def main(args):
    # 1. Load model
    model, tokenizer = ModelUtils.load_model(
        model_path=args.model_path,
        lora_adapters=args.lora_adapters,
        device=args.device,
        quantize_bits=args.quantize_bits,
    )

    # 2. Load dataset
    dataset = DatasetUtils.load_dataset(
        path=args.dataset_path,
        source_format=args.source_format,
        target_format=args.target_format,
        max_samples=args.max_samples
    )

    ppl_input_list = []
    ppl_input_output_list = []

    input_exceed_cnt = 0
    input_output_exceed_cnt = 0

    for sample in tqdm(dataset, desc="🌟 Evaluating PPL"):
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        output = sample.get("output", "").strip()

        if input_text:
            prompt_input = f"{instruction}\n{input_text}"
        else:
            prompt_input = instruction
        prompt_input_output = f"{prompt_input}\n{output}"

        ppl_input = compute_ppl(model, tokenizer, prompt_input, device=args.device)
        ppl_input_output = compute_ppl(model, tokenizer, prompt_input_output, device=args.device)

        ppl_input_list.append(ppl_input)
        ppl_input_output_list.append(ppl_input_output)

        if args.ppl_threshold_input is not None and ppl_input > args.ppl_threshold_input:
            input_exceed_cnt += 1
        if args.ppl_threshold_input_output is not None and ppl_input_output > args.ppl_threshold_input_output:
            input_output_exceed_cnt += 1

    # Filter outliers, only keep valid values for statistics
    valid_ppl_input_list, input_invalid_cnt = filter_valid_ppl(ppl_input_list)
    valid_ppl_input_output_list, input_output_invalid_cnt = filter_valid_ppl(ppl_input_output_list)

    # Calculate averages (prevent division by zero)
    avg_ppl_input = (
        sum(valid_ppl_input_list) / len(valid_ppl_input_list)
        if valid_ppl_input_list else float('nan')
    )
    avg_ppl_input_output = (
        sum(valid_ppl_input_output_list) / len(valid_ppl_input_output_list)
        if valid_ppl_input_output_list else float('nan')
    )

    # Print results
    print("\n📊 PPL Evaluation Results (valid values only):")
    print(f" - Average PPL (prompt only)......: {avg_ppl_input:.4f} (valid samples: {len(valid_ppl_input_list)})")
    print(f" - Average PPL (prompt + output)..: {avg_ppl_input_output:.4f} (valid samples: {len(valid_ppl_input_output_list)})")
    print(f" - Invalid PPL count (prompt only)......: {input_invalid_cnt}")
    print(f" - Invalid PPL count (prompt + output)..: {input_output_invalid_cnt}")

    # Threshold statistics (maintain original logic)
    if args.ppl_threshold_input is not None:
        ratio = input_exceed_cnt / len(ppl_input_list) * 100
        print(f" - PPL > {args.ppl_threshold_input} [prompt only].......: {input_exceed_cnt}/{len(ppl_input_list)} ({ratio:.2f}%)")
    if args.ppl_threshold_input_output is not None:
        ratio = input_output_exceed_cnt / len(ppl_input_output_list) * 100
        print(f" - PPL > {args.ppl_threshold_input_output} [prompt+output]..: {input_output_exceed_cnt}/{len(ppl_input_output_list)} ({ratio:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PPL metrics")
    parser.add_argument("--model_path", type=str, required=True, help="Model path (HuggingFace format or bin)")
    parser.add_argument('--lora_adapters', type=str, nargs='+', default=[], help="List of LoRA adapter paths")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device")
    parser.add_argument("--quantize_bits", type=int, choices=[4, 8, 16, 32], default=None, help="Quantization bits")
    parser.add_argument("--dataset_path", type=str, required=True, help="Alpaca format dataset path")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--source_format", type=str, default="alpaca", help="Original dataset format (e.g., 'firefly', 'alpaca')")
    parser.add_argument("--target_format", type=str, default="alpaca", help="Target dataset format (e.g., 'alpaca')")
    parser.add_argument("--ppl_threshold_input", type=float, default=None, help="PPL threshold (prompt only)")
    parser.add_argument("--ppl_threshold_input_output", type=float, default=None, help="PPL threshold (prompt + output)")
    args = parser.parse_args()

    main(args)
