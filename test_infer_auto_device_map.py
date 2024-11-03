import os

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "facebook/opt-13b"
dummy_model_dir = "dummy_model_dir"


# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map="auto", torch_dtype="auto"
# )
# print(model.config)

config = AutoConfig.from_pretrained(model_id)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print(model)
device_map = infer_auto_device_map(model, no_split_module_classes="LlamaDecoderLayer")

import json

print(json.dumps(device_map))
# model.save_pretrained(dummy_model_dir, torch_dtype="bfloat16", safe_serialization=False)
