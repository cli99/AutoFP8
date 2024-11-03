import os
import tempfile

from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

activation_scheme = "dynamic"

pretrained_model_dir = "llama3.1/databricks/llama3-405b-instruct/"
quantized_model_dir = "llama3.1/databricks/llama3-405b-instruct-fp8-noattn-nolast/"

# pretrained_model_dir = (
#     "llama3.1/databricks/llama3-8b-instruct"
# )
# quantized_model_dir = (
#     "llama3.1/databricks/llama3-8b-instruct-fp8-noattn"
# )

# quantized_model_dir = "pretrained_model_dir + "-FP8-" + activation_scheme"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto_fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme=activation_scheme,
    ignore_patterns=["re:.*lm_head", "re:.*self_attn"],
    # ignore_patterns=["re:.*lm_head", "re:.*self_attn", "re:.*layers.125"],
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config=quantize_config,
    device_map="cpu",
)

model.quantize(examples)
model.save_quantized(quantized_model_dir)

named_modules = list(model.model.named_modules())
for name, linear in named_modules:
    if hasattr(linear, "weight"):
        print(name, linear.weight.device)
