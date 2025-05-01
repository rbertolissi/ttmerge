# Test-Time Model Merging (TTMM)

A library for efficiently selecting and merging expert LoRAs at Test-Time

**[Documentation](https://example.com)**

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

# 1. Load base components
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
encoder = SentenceTransformer("all-mpnet-base-v2")

# 2. Load normalized corpus embeddings that represent expert domains
# Each row corresponds to an expert: row 0 is the normalized embedding vector of expert 0, row 1 is for expert 1, etc.
# Typically, you would load these pre-computed embeddings from a file
corpus_normalised_embeddings = torch.load("path/to/corpus_embeddings.pt")  # Shape: [n_experts, embedding_dim]

# Directory containing numbered subdirectories (0/, 1/, etc.) where expert LoRAs are stored.
# The individual expert LoRAs should be in the format that the PEFT library uses, with each directory containing adapter_config.json and adapter_model.safetensors files.
adapter_location = "path/to/adapters" 

# 3. Initialize the TestTimeMergingModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
merging_model = TestTimeMergingModel(
    corpus=corpus_normalised_embeddings,
    tokenizer=tokenizer,
    encoder=encoder,
    base_model=base_model,
    device=device,
    adapter_location="path/to/adapters"
    verbose=True
)

# 4. Generate text using relevant expert models.
prompt = "Quantum computing is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# The model automatically selects and merges the most relevant expert adapters
generated_ids = merging_model.generate(
    input_ids,
    max_length=512,
    temperature=0.7,
    do_sample=True
)

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

## Development

### CI checks

* The code is auto-formatted using `black .`.
* Static type checks can be run using `pyright`.
* Tests can be run using `pytest test`.

## Citation

```bibtex

```