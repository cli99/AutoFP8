import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch_device = "cuda"
device_map = {
    "transformer.wte": f"{torch_device}:0",
    "transformer.wpe": f"{torch_device}:0",
    "transformer.h.0": "cpu",
    "transformer.h.1": "cpu",
    "transformer.h.2": "cpu",
    "transformer.h.3": "disk",
    "transformer.h.4": "disk",
    "transformer.ln_f": f"{torch_device}:0",
    "lm_head": f"{torch_device}:0",
}

# check_models_equal requires onloaded tensors
model_id = "hf-internal-testing/tiny-random-gpt2"
onloaded_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
inputs = torch.tensor([[1, 2, 3]])
cpu_output = onloaded_model(inputs)[0]

offload_folder = os.path.join(".", "offload")
offloaded_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map=device_map, offload_folder=offload_folder
)
# presaved_output = offloaded_model(inputs)[0]
offloaded_model.save_pretrained(
    "offload-save", max_shard_size="200KB"
)  # model is 1.6MB, max shard size is allocated to cpu by default
saved_model = AutoModelForCausalLM.from_pretrained(
    "offload-save", device_map=device_map
)
postsaved_output = saved_model(inputs)[0]

# assert torch.allclose(cpu_output, presaved_output, atol=1e-4)
# assert torch.allclose(presaved_output, postsaved_output)
