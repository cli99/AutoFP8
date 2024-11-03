import os
import tempfile

from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

# pretrained_model_dir = "blhf/llama-405b-dummy-hf"
pretrained_model_dir = "llama3.1/databricks/llama3-405b-instruct/"
# pretrained_model_dir = "databricks/llama-8b-instruct"
# pretrained_model_dir = "llama3.1/dbrx-medium-instruct"
activation_scheme = "dynamic"
# quantized_model_dir = "pretrained_model_dir + "-FP8-" + activation_scheme"
quantized_model_dir = "llama3.1/databricks/llama3-405b-instruct-fp8/"

device_map = {
    "model.embed_tokens": "cpu",
    "model.layers.0": "cpu",
    "model.layers.1": "cpu",
    "model.layers.2": "cpu",
    "model.layers.3": "cpu",
    "model.layers.4": 1,
    "model.layers.5": 1,
    "model.layers.6": 1,
    "model.layers.7": 1,
    "model.layers.8": 1,
    "model.layers.9": 1,
    "model.layers.10": 2,
    "model.layers.11": 2,
    "model.layers.12": 2,
    "model.layers.13": 2,
    "model.layers.14": 2,
    "model.layers.15": 2,
    "model.layers.16": 3,
    "model.layers.17": 3,
    "model.layers.18": 3,
    "model.layers.19": 3,
    "model.layers.20": 3,
    "model.layers.21": 3,
    "model.layers.22": 4,
    "model.layers.23": 4,
    "model.layers.24": 4,
    "model.layers.25": 4,
    "model.layers.26": 4,
    "model.layers.27": 4,
    "model.layers.28": 5,
    "model.layers.29": 5,
    "model.layers.30": 5,
    "model.layers.31": 5,
    "model.layers.32": 5,
    "model.layers.33": 5,
    "model.layers.34": 6,
    "model.layers.35": 6,
    "model.layers.36": 6,
    "model.layers.37": 6,
    "model.layers.38": 6,
    "model.layers.39": 6,
    "model.layers.40": 7,
    "model.layers.41": 7,
    "model.layers.42": 7,
    "model.layers.43": 7,
    "model.layers.44": 7,
    "model.layers.45": 7,
    "model.layers.46": "cpu",
    "model.layers.47": "cpu",
    "model.layers.48": "cpu",
    "model.layers.49": "cpu",
    "model.layers.50": "cpu",
    "model.layers.51": "cpu",
    "model.layers.52": "cpu",
    "model.layers.53": "cpu",
    "model.layers.54": "cpu",
    "model.layers.55": "cpu",
    "model.layers.56": "cpu",
    "model.layers.57": "cpu",
    "model.layers.58": "cpu",
    "model.layers.59": "cpu",
    "model.layers.60": "cpu",
    "model.layers.61": "cpu",
    "model.layers.62": "cpu",
    "model.layers.63": "cpu",
    "model.layers.64": "cpu",
    "model.layers.65": "cpu",
    "model.layers.66": "cpu",
    "model.layers.67": "cpu",
    "model.layers.68": "cpu",
    "model.layers.69": "cpu",
    "model.layers.70": "cpu",
    "model.layers.71": "cpu",
    "model.layers.72": "cpu",
    "model.layers.73": "cpu",
    "model.layers.74": "cpu",
    "model.layers.75": "cpu",
    "model.layers.76": "cpu",
    "model.layers.77": "cpu",
    "model.layers.78": "cpu",
    "model.layers.79": "cpu",
    "model.layers.80": "cpu",
    "model.layers.81": "cpu",
    "model.layers.82": "cpu",
    "model.layers.83": "cpu",
    "model.layers.84": "cpu",
    "model.layers.85": "cpu",
    "model.layers.86": "cpu",
    "model.layers.87": "cpu",
    "model.layers.88": "cpu",
    "model.layers.89": "cpu",
    "model.layers.90": "cpu",
    "model.layers.91": "cpu",
    "model.layers.92": "cpu",
    "model.layers.93": "cpu",
    "model.layers.94": "cpu",
    "model.layers.95": "cpu",
    "model.layers.96": "cpu",
    "model.layers.97": "cpu",
    "model.layers.98": "cpu",
    "model.layers.99": "cpu",
    "model.layers.100": "cpu",
    "model.layers.101": "cpu",
    "model.layers.102": "cpu",
    "model.layers.103": "cpu",
    "model.layers.104": "cpu",
    "model.layers.105": "cpu",
    "model.layers.106": "cpu",
    "model.layers.107": "cpu",
    "model.layers.108": "cpu",
    "model.layers.109": "cpu",
    "model.layers.110": "cpu",
    "model.layers.111": "cpu",
    "model.layers.112": "cpu",
    "model.layers.113": "cpu",
    "model.layers.114": "cpu",
    "model.layers.115": "cpu",
    "model.layers.116": "cpu",
    "model.layers.117": "cpu",
    "model.layers.118": "cpu",
    "model.layers.119": "cpu",
    "model.layers.120": "cpu",
    "model.layers.121": "cpu",
    "model.layers.122": "cpu",
    "model.layers.123": "cpu",
    "model.layers.124": "cpu",
    "model.layers.125": "cpu",
    "model.norm": "cpu",
    "lm_head": "cpu",
}

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto_fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8", activation_scheme=activation_scheme
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
