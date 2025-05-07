r"""
`ttmerge` is a Python package for efficient expert model merging based on topic similarity.

`ttmerge` implements **Test-Time Model Merging (TTMM)**, a scalable alternative to Test-Time Training (TTT). 
Instead of fine-tuning a model for every prompt, `ttmerge` selects and merges the most relevant expert adapters at inference time using semantic similarity. 
This allows the model to achieve performance close to TTT at a significantly lower computational cost, with minimal overhead.

Source code: [GitHub Repository](https://github.com/rbertolissi/ttmerge)

# Getting Started
## Installation
You can install `ttmerge` from [PyPI](https://pypi.org/project/ttmerge/) via pip:

```bash
pip install ttmerge
```

## Usage Example
Given a [PyTorch](https://pytorch.org) language model and a collection of expert adapters (e.g., LoRA modules), we can use `ttmerge` to dynamically merge these adapters at inference time based on semantic similarity to the input prompt.

You'll need the following components:
- A base language model (`base_model`) such as one from Hugging Face Transformers.
- A `tokenizer` compatible with the base model.
- A sentence-level encoder (e.g., from `sentence-transformers`) to compute prompt and expert embeddings.
- A matrix of **pre-computed, normalized expert embeddings**, typically representing topics or domains. This is a `torch.Tensor` of shape `[n_experts, embedding_dim]`, where each row corresponds to embeddings of an expert.
- A directory of expert adapters (`adapter_location`), structured as numbered subdirectories (`0/`, `1/`, ..., `n-1/`), each containing a LoRA adapter saved using [PEFT](https://github.com/huggingface/peft) format: `adapter_config.json` and `adapter_model.safetensors`.

With these, we can initialize and use the `TestTimeMergingModel`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from ttmerge import TestTimeMergingModel

# Load base components
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
encoder = SentenceTransformer("all-mpnet-base-v2")

# Load expert embeddings representing expert topics/domains
expert_embeddings = torch.load("path/to/expert_embeddings.pt")  # Shape: [n_experts, embedding_dim]

# Path to expert adapters (PEFT format)
adapter_location = "path/to/adapters"

# Initialize test-time merging model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TestTimeMergingModel(
    expert_embeddings=expert_embeddings,
    tokenizer=tokenizer,
    encoder=encoder,
    base_model=base_model,
    device=device,
    adapter_location=adapter_location,
    verbose=True
)
```

Once initialized, the model automatically selects and merges the most relevant expert adapters based on the input:
```python
prompt = "Quantum computing is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate text using merged expert models
output_ids = model.generate(
    input_ids,
    max_length=512,
    temperature=0.7,
    do_sample=True
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

## Citation

```bibtex
% Citation coming soon.
```


"""


from .merging_model import TestTimeMergingModel

__all__ = ["TestTimeMergingModel"]

__version__ = "0.0.3"
__author__ = "Ryo Bertolissi"
__credits__ = "ETH Zurich, Switzerland"
