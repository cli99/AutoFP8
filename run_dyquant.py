from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = (
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
activation_scheme = "dynamic"
quantized_model_dir = pretrained_model_dir + "-fp8"

quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config=quantize_config,
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto_fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")

model.quantize(examples)
model.save_quantized(quantized_model_dir)
