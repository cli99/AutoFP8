from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8-static-ultrachat_2k-test"

# model_id = "meta-llama/Meta-Llama-3-70B"
# quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8-static-ultrachat_2k"

# model_id = "facebook/opt-125m"
# quantized_model_dir = "opt-125m"

pretrained_model_dir = model_id
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
examples = [
    tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds
]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to(
    "cuda"
)

# examples = ["auto_fp8 is an easy-to-use model quantization library"]
# examples = tokenizer(examples, return_tensors="pt").to("cuda:7")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static",
    # ignore_patterns=["re:.*lm_head"],
    ignore_patterns=[],
    # kv_cache_quant_targets=("k_proj", "v_proj"),
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
