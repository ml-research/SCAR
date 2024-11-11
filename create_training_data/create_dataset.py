from datasets import load_dataset, Features, Sequence, Value, Dataset, load_from_disk
from typing import List
import torch
import transformers
from tqdm import tqdm
import multiprocess

multiprocess.set_start_method("spawn", force=True)

from typing import List

from hf_token import HF_TOKEN


class HookedTransformer:
    """Auxilliary class used to extract activations from transformer models."""

    def __init__(self, block: int) -> None:
        self.block = block
        self.site = None
        self.remove_handle = None

        self._features = None

    def register_with(self, model, site="mlp"):
        self.site = site

        if site == "mlp":
            self.remove_handle = model.model.layers[
                self.block
            ].mlp.register_forward_hook(self)
        elif site == "block":
            self.remove_handle = model.model.layers[self.block].register_forward_hook(
                self
            )
        elif site == "attention":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return self

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

    def __call__(self, module, inp, outp) -> None:
        self._features = outp


class Encode:
    def __init__(
        self,
        blocks: List[int],
        model: transformers.models.llama.modeling_llama.LlamaForCausalLM,
        tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model

        self.hooks = {}
        for block in blocks:
            self.hooks[f"Block {block}"] = HookedTransformer(block).register_with(model)

    def encode(self, sentence: dict):
        tokenized = self.tokenizer(
            sentence["text"], return_tensors="pt", truncation=True
        )

        self.model(tokenized["input_ids"].cuda())
        activations = {
            block: hook.pop().detach().cpu() for block, hook in self.hooks.items()
        }

        torch.cuda.empty_cache()

        return {**tokenized, **activations}

    def run(self, dataset: Dataset) -> Dataset:
        ds = {
            "text": [],
            "input_ids": [],
            "Block -1": [],
        }
        for sentence in tqdm(dataset):
            activations = self.encode(sentence)

            ds["text"].append(sentence["text"])
            ds["input_ids"].append(activations["input_ids"])

            ds["Block -1"].append(activations["Block -1"])
            dataset.cleanup_cache_files()

        return Dataset.from_dict(ds)


if __name__ == "__main__":
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        token=HF_TOKEN,
    )
    model = model.half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        token=HF_TOKEN,
    )

    for parameter in model.model.parameters():
        parameter.requires_grad = False

    dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        split="train[:20000]",
        cache_dir="/nfs/scratch_2/ruben/cache",
    )

    dataset.cleanup_cache_files()
    dataset = dataset.remove_columns(["id", "url", "title"])

    torch.cuda.empty_cache()
    model.cuda()
    Enc = Encode([-1], model, tokenizer)

    dataset = Enc.run(dataset)
    dataset.set_format("pt", columns=["input_ids", "Block -1"], output_all_columns=True)
    dataset.save_to_disk("/nfs/scratch_2/ruben/wiki_dataset")
    torch.cuda.empty_cache()

    dataset = load_from_disk("/nfs/scratch_2/ruben/wiki_datase_act")
    tmp = []
    for entry in tqdm(iter(dataset)):
        tmp += list(entry["Block -1"][0])

    dataset = Dataset.from_dict({"Block -1": tmp})
    dataset = dataset.cast(features=Features({"Block -1": Sequence(Value("float16"))}))
    dataset.set_format("pt", columns=["Block -1"], output_all_columns=True)
    dataset.save_to_disk("/nfs/scratch_2/ruben/wiki_dataset_act_fp16")
