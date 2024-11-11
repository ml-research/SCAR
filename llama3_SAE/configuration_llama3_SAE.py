from transformers import LlamaConfig
from typing import List

class LLama3_SAE_Config(LlamaConfig):
    model_type = "llama3_SAE"

    def __init__(
        self,
        base_model_name: str = "",
        hook_block_num: int = 25,
        n_latents: int = 12288,
        n_inputs: int = 4096,
        activation: str = "relu",
        activation_k: int = 64,
        site: str = "mlp",
        tied: bool = False,
        normalize: bool = False,
        mod_features: List[int] = None,
        mod_threshold: List[int] = None,
        mod_replacement: List[int] = None,
        mod_scaling: List[int] = None,
        **kwargs,
    ):

        self.base_model_name = base_model_name
        self.hook_block_num = hook_block_num
        self.n_latents = n_latents
        self.n_inputs = n_inputs
        self.activation = activation
        self.activation_k = activation_k
        self.site = site
        self.tied = tied
        self.normalize = normalize
        self.mod_features = mod_features
        self.mod_threshold = mod_threshold
        self.mod_replacement = mod_replacement
        self.mod_scaling = mod_scaling

        super().__init__(**kwargs)
