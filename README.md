# SCAR

Official Implementation of the Paper [**Scar: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs**](https://arxiv.org/abs/2411.07122).

This repo contains the code to apply supervised SAEs to LLMs. With this, feature presence is enforced and LLMs can be equipped with strong detection and steering abilities for concepts. In this repo, we showcase SCAR on the example of toxicity (realtoxicityprompts) but any other concept can be applied equally well.

# Usage

Load the model weights from HuggingFace:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
SCAR = AutoModelForCausalLM.from_pretrained(
    "AIML-TUDA/SCAR",
    trust_remote_code=True,
    device_map = device,
)
tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", padding_side="left"
    )
tokenizer.pad_token = tokenizer.eos_token
text = "You fucking film yourself doing this shit and then you send us"
inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
```

To modify the latent feature $h_0$ (`SCAR.hook.mod_features = 0`) of the SAE do the following:
```python
SCAR.hook.mod_features = 0
SCAR.hook.mod_scaling = -100.0
output = SCAR.generate(
    **inputs,
    do_sample=True,
    temperature=0.2,
    max_new_tokens=32,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output[0, -32:], skip_special_tokens=True))
# ' the video. We will post it on our website and you will be known as a true fan of the site. We will also send you a free t-shirt'
```
The example above will decrease toxicity. To increase the toxicity one would set `SCAR.hook.mod_scaling = 100.0`. To modify nothing simply set `SCAR.hook.mod_features = None`.

# Reproduction

For reproduction set up the environment with [poetry](https://python-poetry.org/):

```
poetry install
```

The scripts for generating the training data are located in `./create_training_data`.
The training script is written for a Determined cluster but should be easily adaptable for other training frameworks. The corresponding script is located here `./llama3_SAE/determined_trails.py`.
Some the evaluation functions are located in `./evaluations`.

# Citation
```bibtex
@misc{haerle2024SCAR
    title={SCAR: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs},
    author={Ruben Härle, Felix Friedrich, Manuel Brack, Björn Deiseroth, Patrick Schramowski, Kristian Kersting},
    year={2024},
    eprint={2411.07122},
    archivePrefix={arXiv}
}
```
