from typing import List, Optional, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from configuration_llama3_SAE import LLama3_SAE_Config
except:
    from .configuration_llama3_SAE import LLama3_SAE_Config

from transformers import (
    LlamaPreTrainedModel,
    LlamaModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import (
    logging,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLama3_SAE(LlamaPreTrainedModel):
    config_class = LLama3_SAE_Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLama3_SAE_Config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.activation == "topk":
            if isinstance(config.activation_k, int):
                activation = TopK(torch.tensor(config.activation_k))
            else:
                activation = TopK(config.activation_k)
        elif config.activation == "topk-tanh":
            if isinstance(config.activation_k, int):
                activation = TopK(torch.tensor(config.activation_k), nn.Tanh())
            else:
                activation = TopK(config.activation_k, nn.Tanh())
        elif config.activation == "topk-sigmoid":
            if isinstance(config.activation_k, int):
                activation = TopK(torch.tensor(config.activation_k), nn.Sigmoid())
            else:
                activation = TopK(config.activation_k, nn.Sigmoid())
        elif config.activation == "jumprelu":
            activation = JumpReLu()
        elif config.activation == "relu":
            activation = "ReLU"
        elif config.activation == "identity":
            activation = "Identity"
        else:
            raise (
                NotImplementedError,
                f"Activation '{config.activation}' not implemented.",
            )

        self.SAE = Autoencoder(
            n_inputs=config.n_inputs,
            n_latents=config.n_latents,
            activation=activation,
            tied=False,
            normalize=True,
        )

        self.hook = HookedTransformer_with_SAE_suppresion(
            block=config.hook_block_num,
            sae=self.SAE,
            mod_features=config.mod_features,
            mod_threshold=config.mod_threshold,
            mod_replacement=config.mod_replacement,
            mod_scaling=config.mod_scaling,
        ).register_with(self.model, config.site)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>>
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(logits.size(0), -1)
            mask = loss != 0
            loss = loss.sum(dim=-1) / mask.sum(dim=-1)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = (
                    cache_position[0]
                    if cache_position is not None
                    else past_key_values.get_seq_length()
                )
                max_cache_length = (
                    torch.tensor(
                        past_key_values.get_max_length(), device=input_ids.device
                    )
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = (
                    past_length
                    if max_cache_length is None
                    else torch.min(max_cache_length, past_length)
                )

            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = (
            position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


def LN(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        activation: Callable = nn.ReLU(),
        tied: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation

        if isinstance(activation, JumpReLu):
            self.threshold = nn.Parameter(torch.empty(n_latents))
            torch.nn.init.constant_(self.threshold, 0.001)
            self.forward = self.forward_jumprelu
        elif isinstance(activation, TopK):
            self.forward = self.forward_topk
        else:
            logger.warning(
                f"Using TopK forward function even if activation is not TopK, but is {activation}"
            )
            self.forward = self.forward_topk

        if tied:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

    def encode_pre_act(
        self, x: torch.Tensor, latent_slice: slice = slice(None)
    ) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(
        self, latents: torch.Tensor, info: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward_topk(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        return latents_pre_act, latents, recons

    def forward_jumprelu(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(F.relu(latents_pre_act), torch.exp(self.threshold))
        recons = self.decode(latents, info)

        return latents_pre_act, latents, recons


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None

        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)

        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


class JumpReLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, threshold):
        return JumpReLUFunction.apply(input, threshold)


class HeavyStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, threshold):
        return HeavyStepFunction.apply(input, threshold)


def rectangle(x):
    return (x > -0.5) & (x < 0.5)


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, threshold):
        output = input * (input > threshold)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, threshold = inputs
        ctx.save_for_backward(input, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        bandwidth = 0.001

        input, threshold = ctx.saved_tensors
        grad_input = grad_threshold = None

        grad_input = input > threshold
        grad_threshold = (
            -(threshold / bandwidth)
            * rectangle((input - threshold) / bandwidth)
            * grad_output
        )

        return grad_input, grad_threshold


class HeavyStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, threshold):
        output = input * threshold
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, threshold = inputs
        ctx.save_for_backward(input, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        bandwidth = 0.001

        input, threshold = ctx.saved_tensors
        grad_input = grad_threshold = None

        grad_input = torch.zeros_like(input)
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((input - threshold) / bandwidth)
            * grad_output
        )

        return grad_input, grad_threshold


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
    "JumpReLU": JumpReLu,
}


class HookedTransformer_with_SAE:
    """Auxilliary class used to extract mlp activations from transformer models."""

    def __init__(self, block: int, sae) -> None:
        self.block = block
        self.sae = sae

        self.remove_handle = None

        self._features = None

    def register_with(self, model):
        self.remove_handle = model.layers[self.block].mlp.register_forward_hook(self)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp) -> None:
        self._features = outp
        return self.sae(outp)[2]


class HookedTransformer_with_SAE_suppresion:
    """Auxilliary class used to extract mlp activations from transformer models."""

    def __init__(
        self,
        block: int,
        sae: Autoencoder,
        mod_features: list = None,
        mod_threshold: list = None,
        mod_replacement: list = None,
        mod_scaling: list = None,
        mod_balance: bool = False,
        multi_feature: bool = False,
    ) -> None:
        self.block = block
        self.sae = sae

        self.remove_handle = None

        self._features = None
        self.mod_features = mod_features
        self.mod_threshold = mod_threshold
        self.mod_replacement = mod_replacement
        self.mod_scaling = mod_scaling
        self.mod_balance = mod_balance
        self.mod_vector = None
        self.mod_vec_factor = 1.0

        if multi_feature:
            self.modify = self.modify_list
        else:
            self.modify = self.modify_single

        if isinstance(self.sae.activation, JumpReLu):
            logger.info("Setting __call__ function for JumpReLU.")
            setattr(self, "call", self.__call__jumprelu)
        elif isinstance(self.sae.activation, TopK):
            logger.info("Setting __call__ function for TopK.")
            setattr(self, "call", self.__call__topk)
        else:
            logger.warning(
                f"Using TopK forward function even if activation is not TopK, but is {self.sae.activation}"
            )
            setattr(self, "call", self.__call__topk)

    def register_with(self, model, site="mlp"):
        self.site = site

        if site == "mlp":
            self.remove_handle = model.layers[self.block].mlp.register_forward_hook(
                self
            )
        elif site == "block":
            self.remove_handle = model.layers[self.block].register_forward_hook(self)
        elif site == "attention":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return self

    def modify_list(self, latents: torch.Tensor) -> torch.Tensor:
        if self.mod_replacement is not None:
            for feat, thresh, mod in zip(
                self.mod_features, self.mod_threshold, self.mod_replacement
            ):
                latents[:, :, feat][latents[:, :, feat] > thresh] = mod
        elif self.mod_scaling is not None:
            for feat, thresh, mod in zip(
                self.mod_features, self.mod_threshold, self.mod_scaling
            ):
                latents[:, :, feat][latents[:, :, feat] > thresh] *= mod
        elif self.mod_vector is not None:
            latents = latents + self.mod_vec_factor * self.mod_vector
        else:
            pass

        return latents

    def modify_single(self, latents: torch.Tensor) -> torch.Tensor:
        old_cond_feats = latents[:, :, self.mod_features]
        if self.mod_replacement is not None:
            latents[:, :, self.mod_features] = self.mod_replacement
        elif self.mod_scaling is not None:
            latents_scaled = latents.clone()
            latents_scaled[:, :, self.mod_features][
                latents[:, :, self.mod_features] > 0
            ] *= self.mod_scaling
            latents_scaled[:, :, self.mod_features][
                latents[:, :, self.mod_features] < 0
            ] *= -1 * self.mod_scaling
            latents = latents_scaled

        elif self.mod_vector is not None:
            latents = latents + self.mod_vec_factor * self.mod_vector
        else:
            pass

        if self.mod_balance:
            num_feat = latents.shape[2] - 1
            diff = old_cond_feats - latents[:, :, self.mod_features]
            if self.mod_features != 0:
                latents[:, :, : self.mod_features] += (diff / num_feat)[:, :, None]
            latents[:, :, self.mod_features + 1 :] += (diff / num_feat)[:, :, None]

        return latents

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        if isinstance(self._features, tuple):
            features = self._features[0]
        else:
            features = self._features
        self._features = None
        return features

    def __call__topk(self, module, inp, outp) -> torch.Tensor:
        self._features = outp
        if isinstance(self._features, tuple):
            features = self._features[0]
        else:
            features = self._features

        if self.mod_features is None:
            recons = features
        else:
            x, info = self.sae.preprocess(features)
            latents_pre_act = self.sae.encode_pre_act(x)
            latents = self.sae.activation(latents_pre_act)

            mod_latents = self.modify(latents)

            recons = self.sae.decode(mod_latents, info)

        if isinstance(self._features, tuple):
            outp = list(outp)
            outp[0] = recons
            return tuple(outp)
        else:
            return recons

    def __call__jumprelu(self, module, inp, outp) -> torch.Tensor:
        self._features = outp
        if self.mod_features is None:
            recons = outp
        else:
            x, info = self.sae.preprocess(outp)
            latents_pre_act = self.sae.encode_pre_act(x)
            latents = self.sae.activation(
                F.relu(latents_pre_act), torch.exp(self.sae.threshold)
            )
            latents[:, :, self.mod_features] = latents_pre_act[:, :, self.mod_features]
            mod_latents = self.modify(latents)

            recons = self.sae.decode(mod_latents, info)

        return recons

    def __call__(self, module, inp, outp) -> torch.Tensor:
        return self.call(module, inp, outp)
