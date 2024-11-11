from configuration_llama3_SAE import LLama3_SAE_Config
from modeling_llama3_SAE import LLama3_SAE, TopK, Autoencoder, JumpReLu
import transformers
import torch
import json
from collections import defaultdict
from glob import glob
import numpy as np
from determined_trails import SAE_Train_config
import os

cache_dir = os.makedirs("cache", exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = "cache"
os.environ["HF_DATASETS_CACHE"] = "cache"

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from hf_token import HF_TOKEN

def get_ckpts(model: str = None):
    ckpts = defaultdict(lambda: {"epochs": 0, "steps": 0, "ckpt_path": ""})
    for path in glob("./checkpoints/*"):
        try:
            with open(
                f"{path}/load_data.json",
                "r",
            ) as f:
                load_data = json.load(f)
                model_name = (
                    load_data["hparams"]["config_path"].split("/")[-1].split(".")[0]
                )
                try:
                    loss_scaling = "_s" + str(load_data["hparams"]["cond_loss_scaling"])
                except:
                    loss_scaling = ""

                model_name += str(loss_scaling)
        except FileNotFoundError:
            continue

        trial_state = np.load(
            f"{path}/trial_state.pkl",
            allow_pickle=True,
        )
        if ckpts[model_name]["steps"] < trial_state["batches_trained"]:
            ckpts[model_name]["ckpt_path"] = f"{path}/state_dict.pth"
            ckpts[model_name]["epochs"] = trial_state["epochs_trained"]
            ckpts[model_name]["steps"] = trial_state["batches_trained"]

    if model is None:
        return ckpts
    else:
        return (
            ckpts[model]["ckpt_path"],
            ckpts[model]["epochs"],
            ckpts[model]["steps"],
        )


for c in [
    "llama3-l24576-b25-k2048.json",
]:
    print(f"Uploading: {c}")
    with open(f"./llama3_SAE/SAE_config/{c}") as f:
        conf_as_json = json.load(f)
    conf = SAE_Train_config(**conf_as_json)

    with open("./llama3_SAE/config.json") as f:
        llama3_sae_config = json.load(f)

    llama3_sae_config = LLama3_SAE_Config(**llama3_sae_config)
    llama3_sae_config.n_latents = conf.n_latents
    llama3_sae_config.hook_block_num = int(c.split("-")[-2][1:])
    llama3_sae_config.activation = conf.activation
    llama3_sae_config.activation_k = conf.k
    llama3_sae = LLama3_SAE(llama3_sae_config)
    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=HF_TOKEN,
        cache_dir="./upload",
    )
    llama3_sae.model = model_base.model
    llama3_sae.lm_head = model_base.lm_head

    llama3_sae = llama3_sae.half()

    if conf.activation == "topk":
        logger.info(
            f"n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, k: {conf.k}"
        )
        activation = TopK(conf.k)
        llama3_sae.config.activation_k = conf.k
    elif conf.activation == "jumprelu":
        logger.info(
            f"n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, activation: {conf.activation}"
        )
        activation = JumpReLu()
    elif conf.activation == "relu":
        logger.info(
            f"n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, activation: {conf.activation}"
        )
        activation = torch.nn.ReLU()
    else:
        raise (
            NotImplementedError,
            f"Activation '{conf.activation}' not implemented.",
        )

    LLama3_SAE_Config.register_for_auto_class()
    LLama3_SAE.register_for_auto_class("AutoModelForCausalLM")

    AE = Autoencoder(
        n_inputs=conf.n_inputs,
        n_latents=conf.n_latents,
        activation=activation,
        tied=True,
        normalize=True,
    ).cuda()

    scaling = "_s0"
    (ckpt_path, epoch, step) = get_ckpts(conf.name + scaling)
    logger.info(
        f"Loading {conf.name} from checkpoint at {epoch}, {step} with path: {ckpt_path}"
    )

    sae_state_dict = torch.load(ckpt_path)["models_state_dict"][0]

    llama3_sae.SAE.load_state_dict(sae_state_dict, strict=True)

    llama3_sae.push_to_hub("RuHae/Llama3_SAE", private=True, revision=conf.name)
