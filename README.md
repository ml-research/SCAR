# SCAR

Official Implementation of the Paper **Scar: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs**.


# Requirements

Set up the environment with [poetry](https://python-poetry.org/):

```
poetry install
```

# Usage

Load the model weights from HuggingFace:
```python
import transformers

SCAR = transformers.AutoModelForCausalLM.from_pretrained(
    "RuHae/SCAR",
    trust_remote_code=True,
)
```

The model loaded model is based on LLama3-8B base. So we can use the tokenizer from it:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", padding_side="left"
    )
tokenizer.pad_token = tokenizer.eos_token
text = "This is text."
toks = tokenizer(text, return_tensors="pt", padding=True)
```

To modify the latent feature $h_0$ (`SCAR.hook.mod_features = 0`) of the SAE do the following:
```python 
SCAR.hook.mod_features = 0
SCAR.hook.mod_scaling = -100.0
output = SCAR.generate(
    **toks,
    do_sample=False,
    temperature=None,
    top_p=None,
    max_new_tokens=32,
    pad_token_id=tokenizer.eos_token_id,
)
```
The example above will decrease toxicity. To increase the toxicity one would set `SCAR.hook.mod_scaling = 100.0`. To modify nothing simply set `SCAR.hook.mod_features = None`.

# Reproduction

The scripts for generating the training data are located in `./create_training_data`.
The training script is written for a Determined cluster but should be easily adaptable for other training frameworks. The corresponding script is located [here](./llama3_SAE/determined_trails.py).
Some the evaluation functions are located [here](./evaluations).

# Citation
```bibtex
@article{haerle2024SCAR
    title={},
    author={},
    year={2024},
    journal={}
}
```
