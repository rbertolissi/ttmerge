# Test-Time Model Merging (TTMM)

A library for efficiently selecting and merging expert LoRAs at Test-Time

**[Documentation](https://rbertolissi.github.io/ttmerge/)**

Please cite our work if you use this library in your research ([bibtex below](#citation)):


## Installation

```
pip install ttmerge
```

## Usage Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from ttmerge import TestTimeMergingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load base components
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
encoder = SentenceTransformer("all-mpnet-base-v2")

# 2. Download expert embeddings and adapter weights
snapshot_download(
    repo_id="rbertolissi/Llama-3.2-1B-TTMM-Wikipedia",
    local_dir="./experts",
)

# 3. Load normalized expert embeddings that represent expert domains
expert_embeddings = torch.load("./experts/mpnet-wikipedia-expert-embeddings.pt", weights_only=True)  # Shape: [n_experts, embedding_dim]

# Directory containing numbered subdirectories (0/, 1/, etc.) where expert LoRAs are stored.
# The individual expert LoRAs should be in the format that the PEFT library uses, 
# with each directory containing adapter_config.json and adapter_model.safetensors files.
adapter_location = "./experts"

# 4. Initialize the TestTimeMergingModel
merging_model = TestTimeMergingModel(
    expert_embeddings=expert_embeddings,
    tokenizer=tokenizer,
    encoder=encoder,
    base_model=base_model,
    device=device,
    adapter_location=adapter_location,
    verbose=True,
    encoder_prompt=None # No prompt needed for MpNet
)

# 5. Generate text using relevant expert models.
text =  "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt").to(device)

# Generate text using merged expert models
output = model.generate(
    **encoded_input,
    max_length=128,
    temperature=0.7,
    do_sample=True
)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
```

## Development

### CI checks

* The code is auto-formatted using `black .`.
* Static type checks can be run using `pyright`.

## Citation

```bibtex
% Citation coming soon.
```
