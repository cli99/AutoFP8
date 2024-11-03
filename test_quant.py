import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig


def per_tensor_fp_quantize(input: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    min_val, max_val = input.aminmax()

    amax = torch.maximum(max_val.abs(), min_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (input * scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    inv_scale = scale.float().reciprocal()
    return qweight, inv_scale


activation_scheme = "dynamic"
pretrained_model_dir = "hf-internal-testing/tiny-random-gpt2"
quantized_model_dir = "test-out"

torch_device = "cuda"
device_map = {
    "transformer.wte": f"cpu",
    "transformer.wpe": f"cpu",
    "transformer.h.0": "cpu",
    "transformer.h.1": "cpu",
    "transformer.h.2": "cpu",
    "transformer.h.3": "disk",
    "transformer.h.4": "disk",
    "transformer.ln_f": f"cpu",
    "lm_head": f"cpu",
}

quantize_config = BaseQuantizeConfig(
    quant_method="fp8", activation_scheme=activation_scheme
)

with tempfile.TemporaryDirectory() as tmp_dir:
    offload_folder = os.path.join(tmp_dir, "offload")
    model = AutoFP8ForCausalLM.from_pretrained(
        pretrained_model_dir,
        quantize_config=quantize_config,
        device_map=device_map,
        offload_folder=offload_folder,
    )

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto_fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")


inputs = torch.tensor([[1, 2, 3]])
presaved_output = model.model(inputs)[0]

model.quantize(examples)
named_modules = list(model.model.named_modules())
for name, linear in named_modules:
    if hasattr(linear, "weight"):
        print(name, linear.weight.device)

for name, param in model.named_parameters():
    print(name, param.device)

model.save_quantized(quantized_model_dir)
