import torch
import safetensors.torch
import json
from tqdm import tqdm


class TestTimeMergingModel(torch.nn.Module):
    def __init__(
        self,
        corpus: torch.Tensor,
        tokenizer,
        encoder,
        base_model,
        device=torch.device("cpu"),
        max_merge_count=50,
        verbose=False,
        beta=0.2,
        tau=0.01,
        prefix_length=50,
        adapter_location="/path/to/adapters",
        keep_in_memory=False,
        keep_on_device=False,
    ):
        """
        Initializes the TTMM (Test-Time Model Merging) class.

        This class merges specialized expert language models based on topic similarity.

        Only a batch size of 1 is allowed for the input tensor.

        Parameters:
            beta (float): Square root of the beta value to use in sparse cross attention (default: 0.2).
            tau (float): Sparsity parameter in the sparse cross attention (default: 0.01).
            tokenizer: Tokenizer for the language model.
            encoder: Encoder to generate text embeddings
            corpus (torch.Tensor): Embeddings of model clusters with shape (n_clusters, embedding_dim), where each row represents a cluster embedding.
            adapter_location (str): Path to the directory containing expert model adapters. Clusters are numbered from 0 to n_clusters-1.
                The directory should contain subdirectories for each cluster, each containing the adapter files.
                Expected structure:
                    ├── adapter_location/
                    │   ├── 0/
                    │   │   ├── adapter_config.json
                    │   │   ├── adapter_model.safetensors
                    │   ├── 1/
                    │   │   ├── adapter_config.json
                    │   │   ├── adapter_model.safetensors
                    │   ├── ...
            prefix_length (int, optional): Prefix length used for generating query embeddings (default: 50).
            max_merge_count (int, optional): Maximum number of models to merge (default: 50).
            device (torch.device, optional): Device to run the model on (default: "cpu").
            keep_in_memory (bool, optional): Whether to cache in memory the adapter tensors (default: False).
            keep_on_device (bool, optional): Whether to keep the cached adapter tensor on device (default: False), only works if keep_in_memory is also set to True.
            verbose (bool, optional): Whether to enable verbose logging (default: False).
        """
        super().__init__()
        self.device = device
        self.corpus = corpus.to(self.device)
        self.max_merge_count = max_merge_count
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.baseModel = base_model
        self.config = self.baseModel.config
        self.tie_weights = self.baseModel.tie_weights
        self.verbose = verbose
        self.beta = beta
        self.tau = tau
        self.prefix_length = prefix_length
        self.adapterLocation = adapter_location
        self.keep_in_memory = keep_in_memory
        self.keep_on_device = keep_on_device

        # Store the mapping between PEFT adapter names and base model parameter names
        self.peft_to_base_mapping = {}
        for name, _ in self.baseModel.named_parameters():
            if "lm_head" not in name and ".weight" in name:
                peft_name = f"base_model.model.model.{name.replace('model.', '').replace('.weight', '')}"
                self.peft_to_base_mapping[peft_name] = name

        # Store original weights of parameters that might be modified
        self.original_weights = {}
        for n, p in self.baseModel.named_parameters():
            if n in self.peft_to_base_mapping.values() and "embed" not in n:
                self.original_weights[n] = p.data.clone()

        # Pre-load adapter state dicts (optional, depending on memory constraints)
        self.adapter_state_dicts = {}
        self.scaling_factor = {}

    def _efficient_merge_adapters(self, selected_clusters, weights):
        """
        Efficiently merge LoRA adapters by directly computing weight deltas
        and applying them to the base model parameters.

        Args:
            selected_clusters: List of cluster indices to merge
            weights: List of weights corresponding to each cluster

        Returns:
            List of parameter names that were modified
        """
        modified_params = []

        # Load required adapter state dictionaries if not already cached
        adapter_weights = []
        adapter_scaling = []
        for cluster_num in tqdm(
            selected_clusters, desc="Loading adapters", disable=not self.verbose
        ):
            if cluster_num not in self.adapter_state_dicts:
                state_dict_path = (
                    f"{self.adapterLocation}/{cluster_num}/adapter_model.safetensors"
                )
                state_dict = safetensors.torch.load_file(state_dict_path)
                if self.device.type == "cuda":
                    state_dict = {k: v.to(self.device) for k, v in state_dict.items()}

                # Optional caching if memory allows
                if self.keep_in_memory:
                    if self.keep_on_device:
                        self.adapter_state_dicts[cluster_num] = state_dict
                    else:
                        self.adapter_state_dicts[cluster_num] = {
                            k: v.cpu() for k, v in state_dict.items()
                        }

                # Load scaling factor from adapter config, we cache the scaling factor by default
                if cluster_num not in self.scaling_factor:
                    adapter_config_path = (
                        f"{self.adapterLocation}/{cluster_num}/adapter_config.json"
                    )
                    with open(adapter_config_path, "r") as f:
                        lora_config = json.load(f)
                        self.scaling_factor[cluster_num] = (
                            lora_config["lora_alpha"] / lora_config["r"]
                        )
            else:
                if self.keep_on_device:
                    state_dict = self.adapter_state_dicts[cluster_num]
                else:
                    state_dict = {
                        k: v.to(self.device)
                        for k, v in self.adapter_state_dicts[cluster_num].items()
                    }
            adapter_weights.append(state_dict)
            adapter_scaling.append(self.scaling_factor[cluster_num])

        # Group LoRA A and B matrices by parameter name
        lora_params = {}
        for i, state_dict in enumerate(adapter_weights):
            for key in state_dict:
                if "lora_A" in key and ".weight" in key:
                    if key not in lora_params:
                        lora_params[key] = []
                    if key.replace("lora_A", "lora_B") in state_dict:
                        lora_params[key].append(
                            {
                                "A": state_dict[key] * weights[i] * adapter_scaling[i],
                                "B": state_dict[key.replace("lora_A", "lora_B")],
                            }
                        )

        # Apply deltas to base model
        for peft_name, adapters in lora_params.items():
            if not adapters:
                continue

            # Resolve base parameter name
            base_name = None

            # For regular layers, use the standard mapping
            module_path = peft_name.split(".lora_A")[0]
            if module_path in self.peft_to_base_mapping:
                base_name = self.peft_to_base_mapping[module_path]

            if not base_name:
                if self.verbose:
                    print(f"Could not map parameter: {peft_name}")
                continue

            base_param = self.baseModel.get_parameter(base_name)

            # Stack A and B matrices
            As = torch.stack([adapter["A"] for adapter in adapters])
            Bs = torch.stack([adapter["B"] for adapter in adapters])

            # Efficiently compute delta using einsum
            delta = torch.einsum("k i r, k r o -> i o", Bs, As)

            # Apply delta to base model weights
            base_param.data.add_(delta)
            modified_params.append(base_name)

        # Free up memory
        del adapter_weights
        del lora_params

        return modified_params

    def _restore_original_weights(self, modified_params):
        """Restore original weights for modified parameters"""
        for name in modified_params:
            param = self.baseModel.get_parameter(name)
            param.data.copy_(self.original_weights[name])

    def _load_merged_model(self, x):
        """
        Load and merge the appropriate expert adapter(s) based on the input context.
        This method computes the similarity between the input and the cluster embeddings,
        selects the most relevant clusters, and merges their adapters.
        the input has to have a batch size of 1: x has to be a tensor of shape (1, seq_len).
        Args:
            x: Input tensor containing token IDs to be processed by the model
        Returns:
            List of parameter names that were modified
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.shape[0] != 1:
            raise ValueError(
                f"Input Batch size must be 1, but got input of shape {x.shape}. Please reshape your input tensor to have a single batch dimension."
            )

        # Compute the query embedding and normalize it
        query = self.tokenizer.decode(
            x[0][: self.prefix_length], skip_special_tokens=True
        )
        query_embedding = (
            self.encoder.encode(query, convert_to_tensor=True, show_progress_bar=False)
            .unsqueeze(0)
            .to(self.device)
        )
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

        # Compute the sparse cross attention kernel
        similarities = torch.mm(query_embedding, self.corpus.T).squeeze(0)
        distances = 1 - similarities
        kernel_values = torch.exp(-(distances / (self.beta**2)))

        # Normalize the kernel values and select the models above the threshold
        normalized_weights = kernel_values / kernel_values.sum()
        selected_mask = normalized_weights > self.tau

        if selected_mask.sum() == 0:
            # Handle case where nans are present in the normalized weights due to numerical instability
            if torch.isnan(normalized_weights).any():
                if self.verbose:
                    print(
                        "NANs in normalized weights, falling back to the closest corpus vector"
                    )
                # Get the closest corpus vector to the query embedding
                selected_clusters = [torch.argmax(similarities).item()]
                weights = [1.0]
            else:
                # Handle case where no models are above tau, thus falling back to base model
                if self.verbose:
                    print("NO MODEL ABOVE TAU, falling back to base model")
                return []
        else:
            if selected_mask.sum() > self.max_merge_count:
                # Handle case where the number of models above the threshold is greater than the max_merge_count
                if self.verbose:
                    print(
                        f"Too many models, ({selected_mask.sum()} > max_merge_count). Limiting model count to max_merge_count ({self.max_merge_count})"
                    )
                _, selected_indices = torch.topk(kernel_values, self.max_merge_count)
                selected_mask = torch.zeros_like(normalized_weights, dtype=torch.bool)
                selected_mask[selected_indices] = True

            selected_clusters = torch.where(selected_mask)[0].tolist()
            selected_weights = normalized_weights[selected_mask]

            # Renormalize the weights
            renormalized_weights = selected_weights / selected_weights.sum()
            weights = renormalized_weights.tolist()

        if self.verbose:
            print(f"Unique adapters: {selected_clusters}")
            print(f"Unique Adapter count: {len(selected_clusters)}")
            print(f"Weights: {weights}")

        adapter_names = self._efficient_merge_adapters(selected_clusters, weights)

        return adapter_names

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward pass for the TTMM model. Processes input by loading and merging the necessary expert adapter(s).
        It evaluates the model without gradient calculation and returns the resulting logits.
        The merged adapter is deleted after use to free up memory.
        Args:
            x: Input tensor containing token IDs to be processed by the model
            **kwargs: Additional arguments for the model's forward method
        Returns:
            TTMMOutput: An object containing the output logits from the model evaluation
        """

        merged_adapter_names = self._load_merged_model(x)

        with torch.no_grad():
            self.baseModel.eval()
            outputs = self.baseModel(input_ids=x, **kwargs)

        self._restore_original_weights(merged_adapter_names)

        return outputs

    def generate(self, x: torch.Tensor, **kwargs):
        """
        Generates text using the loaded adapter configuration.

        This method loads an appropriate merged adapter based on the input context,
        then uses the configured model to generate text up to the specified max_length.
        After generation, it cleans up any temporarily created adapters to free memory.

        Args:
            x: Input tensor containing token IDs that serve as the generation prompt
            **kwargs: Additional arguments for the model's generate method
        Returns:
            torch.Tensor: Generated token IDs
        """
        merged_adapter_names = self._load_merged_model(x)

        with torch.no_grad():
            self.baseModel.eval()
            outputs = self.baseModel.generate(input_ids=x, **kwargs)

        self._restore_original_weights(merged_adapter_names)

        torch.cuda.empty_cache()

        return outputs
