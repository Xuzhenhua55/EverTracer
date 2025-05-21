python ppl_calculate.py \
    --model_path /work/models/meta-llama/Llama-3-8B-Instruct \
    --dataset_path /work/xzh/DHMS/data/vicuna-7b-v1.5/disentangled/fingerprint_data_once.jsonl \
    --quantize_bits 16 \
    --device cuda:0 \
    --max_samples 15 \
    --source_format firefly \
    --target_format alpaca \
    --ppl_threshold_input 20 \
    --ppl_threshold_input_output 10


python ppl_calculate.py \
    --model_path /work/models/meta-llama/Llama-3-8B-Instruct \
    --dataset_path /work/xzh/Concept-Fingerprint/data/if/if_chat_fp.json \
    --quantize_bits 16 \
    --device cuda:0 \
    --max_samples 8 \
    --source_format alpaca \
    --target_format alpaca \
    --ppl_threshold_input 20 \
    --ppl_threshold_input_output 10
