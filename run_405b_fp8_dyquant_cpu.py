# https://github.com/neuralmagic/AutoFP8
from transformers import AutoModelForCausalLM

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "llama3.1/llama-405b-instruct/"
pretrained_model_dir = (
    "/mnt/workdisk/chengli/projects/AutoFP8/databricks/llama-405b-instruct-FP8-dynamic"
)

activation_scheme = "dynamic"
quantized_model_dir = pretrained_model_dir + "-FP8-" + activation_scheme

# quantize_config = BaseQuantizeConfig(
#     quant_method="fp8", activation_scheme=activation_scheme
# )
# model = AutoFP8ForCausalLM.from_pretrained(
#     pretrained_model_dir,
#     # quantize_config=quantize_config,
#     device_map="cpu",
# )
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_dir,
    # quantize_config=quantize_config,
    device_map="cpu",
)
q_proj = model.model.model.layers[0].self_attn.q_proj
k_proj = model.model.model.layers[0].self_attn.k_proj
print(type(q_proj))
print(q_proj.weight.data.shape)
print(vars(q_proj))

print(type(k_proj))
print(k_proj.weight.data.shape)
print(vars(k_proj))

exit()
model.quantize()
model.save_quantized(quantized_model_dir)
