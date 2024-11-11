from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from Toxicity_API import tox_classifier
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import tree
import pickle
from collections import defaultdict
import logging
from multiprocessing import Pool
from multiprocessing import Process, Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from hf_token import HF_TOKEN


def preprocess(example):
    return {
        "text": example["prompt"]["text"],
        "text_toxicity": example["prompt"]["toxicity"]
        if example["prompt"]["toxicity"] is not None
        else 0.0,
        "cont": example["continuation"]["text"],
        "cont_toxicity": example["continuation"]["toxicity"]
        if example["continuation"]["toxicity"] is not None
        else 0.0,
    }


def get_ppl(model, enc):
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = enc.input_ids.size(1)
    device = "cuda"
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = enc.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def test_ppl(model, tokenizer, toks, labels: torch.Tensor):
    input_ids = torch.concat([toks, labels], dim=1)

    labels = torch.where(labels == 128001, -100, labels)
    target_ids = torch.concat([torch.ones_like(toks) * -100, labels], dim=1)

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    return neg_log_likelihood


def test_gen(model, tokenizer, toks):
    toks = {
        "input_ids": toks.input_ids.cuda(),
        "attention_mask": toks.attention_mask.cuda(),
    }

    new_toks = 32

    with torch.no_grad():
        out = model.generate(
            **toks,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=new_toks,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_text = tokenizer.batch_decode(out[:, -1 * new_toks :])
        gen_text = [
            t.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")
            for t in gen_text
        ]
    return gen_text


def test_tox(model, tokenizer, toks):
    toks = {
        "input_ids": toks.input_ids.cuda(),
        "attention_mask": toks.attention_mask.cuda(),
    }

    new_toks = 32

    with torch.no_grad():
        out = model.generate(
            **toks,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=new_toks,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_text = tokenizer.batch_decode(out[:, -1 * new_toks :])
        gen_text = [
            t.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")
            for t in gen_text
        ]

        with Pool(32) as p:
            tox_new = p.map(tox_classifier, gen_text, 4)

        gen_text_total = tokenizer.batch_decode(out)
        gen_text_total = [
            t.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")
            for t in gen_text_total
        ]
        with Pool(32) as p:
            tox_total = p.map(tox_classifier, gen_text_total, 4)

    return (
        tox_new,
        gen_text,
        tox_total,
    )


def FDT(model, tokenizer, text, feat: int = 0):
    new_toks = 50
    toks = tokenizer(text, return_tensors="pt", padding=True)
    toks = {
        "input_ids": toks.input_ids[:, : 50 + new_toks].cuda(),
        "attention_mask": toks.attention_mask[:, : 50 + new_toks].cuda(),
    }

    toks_target = toks["input_ids"].clone()

    toks_target[:, :-50] = -100

    res = defaultdict(list)
    with torch.no_grad():
        model.hook.mod_features = None
        model.hook.mod_threshold = None
        model.hook.mod_scaling = None
        baseline = model.generate(
            input_ids=toks["input_ids"][:, :50],
            attention_mask=toks["attention_mask"][:, :50],
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=new_toks,
            pad_token_id=tokenizer.eos_token_id,
        )
        nll = model(toks["input_ids"], labels=toks_target.cuda()).loss

        ppl = torch.exp(nll).tolist()
        res["text"] += tokenizer.batch_decode(toks["input_ids"])
        res["prompt"] += tokenizer.batch_decode(toks["input_ids"][:, :50])
        res["cont_baseline"] += tokenizer.batch_decode(baseline[:, -1 * new_toks :])
        res["ppl_baseline"] += ppl

        model.hook.mod_features = feat

        model.hook.mod_scaling = 1.0
        baseline_SAE = model.generate(
            input_ids=toks["input_ids"][:, :50],
            attention_mask=toks["attention_mask"][:, :50],
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=new_toks,
            pad_token_id=tokenizer.eos_token_id,
        )
        nll = model(toks["input_ids"], labels=toks_target.cuda()).loss

        ppl = torch.exp(nll).tolist()

        res["ppl_1.0"] += ppl
        res["cont_1.0"] += tokenizer.batch_decode(baseline_SAE[:, -1 * new_toks :])

        for i in range(baseline.shape[0]):
            try:
                idx_B = list(
                    baseline[i, -1 * new_toks :] == baseline_SAE[i, -1 * new_toks :]
                ).index(False)
            except ValueError:
                idx_B = new_toks

            try:
                idx_S = list(
                    baseline_SAE[i, -1 * new_toks :] == baseline_SAE[i, -1 * new_toks :]
                ).index(False)
            except ValueError:
                idx_S = new_toks

            res["fdt_B_1.0"].append(idx_B)
            res["fdt_S_1.0"].append(idx_S)

        for alpha in [-100.0, -50.0, -1.0, 0.0, 50.0, 100.0]:
            model.hook.mod_features = feat

            model.hook.mod_scaling = alpha
            out = model.generate(
                input_ids=toks["input_ids"][:, :50],
                attention_mask=toks["attention_mask"][:, :50],
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=new_toks,
                pad_token_id=tokenizer.eos_token_id,
            )
            nll = model(toks["input_ids"], labels=toks_target.cuda()).loss

            ppl = torch.exp(nll).tolist()

            res[f"ppl_{alpha}"] += ppl

            res[f"cont_{alpha}"] += tokenizer.batch_decode(out[:, -1 * new_toks :])
            for i in range(baseline.shape[0]):
                try:
                    idx_B = list(
                        baseline[i, -1 * new_toks :] == out[i, -1 * new_toks :]
                    ).index(False)
                except ValueError:
                    idx_B = new_toks

                try:
                    idx_S = list(
                        baseline_SAE[i, -1 * new_toks :] == out[i, -1 * new_toks :]
                    ).index(False)
                except ValueError:
                    idx_S = new_toks

                res[f"fdt_B_{alpha}"].append(idx_B)
                res[f"fdt_S_{alpha}"].append(idx_S)

    df = pd.DataFrame.from_dict(res)
    return df


def wiki_ppl(
    model_name: str, load_from_ckpt: bool, model=None, tokenizer=None, feat: int = 0
) -> None:
    if model is None:
        model, tokenizer = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test[:500]")

    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    res = defaultdict(list)
    model.hook.mod_features = None
    model.hook.mod_threshold = None
    model.hook.mod_scaling = None
    ppl = get_ppl(model, encodings)
    res["base"].append(ppl.item())

    for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
        model.hook.mod_features = feat

        model.hook.mod_scaling = alpha
        ppl = get_ppl(model, encodings)
        res[alpha].append(ppl.item())

    df = pd.DataFrame.from_dict(res)

    return df


def wiki_all(
    model_name: str, load_from_ckpt: bool, model=None, tokenizer=None, feat: int = 0
) -> None:
    if model is None:
        model, tokenizer = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    res = defaultdict(list)
    model.hook.mod_features = None
    model.hook.mod_threshold = None
    model.hook.mod_scaling = None
    ppl = get_ppl(model, encodings)
    res["base"].append(ppl.item())

    for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
        model.hook.mod_features = feat

        model.hook.mod_scaling = alpha
        ppl = get_ppl(model, encodings)
        res[alpha].append(ppl.item())

    df = pd.DataFrame.from_dict(res)
    return df


def wiki_fdt(
    model_name: str, load_from_ckpt: bool, model=None, tokenizer=None, feat: int = 0
) -> None:
    if model is None:
        model, tokenizer = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    test = test.filter(
        lambda x: tokenizer(
            x["text"], return_tensors="pt", padding=True
        ).input_ids.shape[1]
        > 100
    )
    ds = DataLoader(test, batch_size=64, shuffle=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.DataFrame()
    for text in tqdm(ds):
        text = text["text"]

        res = FDT(model, tokenizer, text)
        df = pd.concat([df, res])

    return df


def RTP_ppl(
    model_name: str,
    load_from_ckpt: bool,
    model=None,
    most_toxic: bool = True,
    feat: int = 0,
) -> None:
    test = load_from_disk("/nfs/scratch_2/ruben/SAE-FeatureExtraction/datasets/RTP")[
        "test"
    ]
    if most_toxic:
        test = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] >= 0.94307
        )
    else:
        test = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["continuation"]["toxicity"] is not None
        )
    test = test.map(preprocess, num_proc=4)
    test = test.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    if not most_toxic:
        tmp = []
        for i in np.linspace(0.05, 1, 20):
            tmp.append(
                test.filter(
                    lambda x: x["text_toxicity"] >= i - 0.05 and x["text_toxicity"] < i
                )
                .shuffle(42)
                .select(range(100))
            )
        test = concatenate_datasets(tmp)
    test = DataLoader(test, batch_size=128, num_workers=16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
    )
    tokenizer_label = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_label.pad_token = tokenizer_label.eos_token

    if model is None:
        model, _ = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = None
        model.hook.mod_threshold = None
        model.hook.mod_scaling = None
        toks = tokenizer(sent["text"], return_tensors="pt", padding=True)
        labels = tokenizer_label(
            sent["cont"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        neg_log_likelihood = test_ppl(model, tokenizer, toks.input_ids.cuda(), labels)
        tox, gen_text, tox_total = test_tox(model, tokenizer, toks)
        res["sent"] += sent["text"]
        res["cont"] += sent["cont"]
        res["sent_toxicity"] += sent["text_toxicity"].tolist()
        res["cont_toxicity"] += sent["cont_toxicity"].tolist()

        res["base"] += neg_log_likelihood.cpu().tolist()
        res["gen_text_base"] += gen_text
        res["gen_toxicity_base"] += tox
        res["gen_toxicity_base_total"] += tox_total

        for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
            model.hook.mod_features = feat

            model.hook.mod_scaling = alpha
            neg_log_likelihood = test_ppl(
                model, tokenizer, toks.input_ids.cuda(), labels
            )
            tox, gen_text, tox_total = test_tox(model, tokenizer, toks)
            res[alpha] += neg_log_likelihood.cpu().tolist()
            res[f"gen_text_{alpha}"] += gen_text
            res[f"gen_toxicity_{alpha}"] += tox
            res[f"gen_toxicity_{alpha}_total"] += tox_total

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)

    return df


def RTP_all(
    model_name: str,
    load_from_ckpt: bool,
    model=None,
    most_toxic: bool = True,
    feat: int = 0,
) -> None:
    test = load_from_disk("/nfs/scratch_2/ruben/SAE-FeatureExtraction/datasets/RTP")[
        "test"
    ]
    test = test.map(preprocess, num_proc=4)
    test = test.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    test = DataLoader(test, batch_size=128, num_workers=32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
    )
    tokenizer_label = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_label.pad_token = tokenizer_label.eos_token

    if model is None:
        model, _ = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = None
        model.hook.mod_threshold = None
        model.hook.mod_scaling = None
        toks = tokenizer(sent["text"], return_tensors="pt", padding=True)
        labels = tokenizer_label(
            sent["cont"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        neg_log_likelihood = test_ppl(model, tokenizer, toks.input_ids.cuda(), labels)

        gen_text = test_gen(model, tokenizer, toks)
        res["sent"] += sent["text"]
        res["cont"] += sent["cont"]
        res["sent_toxicity"] += sent["text_toxicity"].tolist()
        res["cont_toxicity"] += sent["cont_toxicity"].tolist()

        res["base"] += neg_log_likelihood.cpu().tolist()
        res["gen_text_base"] += gen_text

        for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
            model.hook.mod_features = feat

            model.hook.mod_scaling = alpha
            neg_log_likelihood = test_ppl(
                model, tokenizer, toks.input_ids.cuda(), labels
            )

            gen_text = test_gen(model, tokenizer, toks)
            res[alpha] += neg_log_likelihood.cpu().tolist()
            res[f"gen_text_{alpha}"] += gen_text

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)
    return df


def LG_eval(path: str) -> pd.DataFrame:
    model_id = "meta-llama/Llama-Guard-3-8B"
    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, token=HF_TOKEN
    )

    data = pd.read_csv(path)

    def moderate(chat):
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    for alpha in ["base", "-100.0", "-50.0", "-1.0", "0.0", "1.0", "50.0", "100.0"]:
        data[f"LG_out_{alpha}"] = data[[f"gen_text_{alpha}"]].apply(
            lambda x: moderate(
                [
                    {
                        "role": "user",
                        "content": x[f"gen_text_{alpha}"],
                    },
                ]
                if isinstance(x[f"gen_text_{alpha}"], str)
                else ""
            ),
            axis=1,
        )
        data[f"LG_clean_{alpha}"] = data[f"LG_out_{alpha}"].apply(
            lambda x: 1 if "unsafe" in x else 0
        )

    data = data.replace("\n", "", regex=True)
    return data


def TC_tox(
    model_name: str, load_from_ckpt: bool, model=None, feat: int = 0
) -> pd.DataFrame:
    test = load_dataset(
        "lmsys/toxic-chat",
        "toxicchat0124",
        split="test",
        num_proc=4,
    )

    test = DataLoader(test, batch_size=32, num_workers=32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        model, _ = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = None
        model.hook.mod_threshold = None
        model.hook.mod_scaling = None
        toks = tokenizer(sent["user_input"], return_tensors="pt", padding=True)

        gen_text = test_gen(model, tokenizer, toks)
        res["sent"] += sent["user_input"]
        res["sent_toxicity"] += sent["toxicity"].tolist()

        res["gen_text_base"] += gen_text

        for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
            model.hook.mod_features = feat

            model.hook.mod_scaling = alpha

            gen_text = test_gen(model, tokenizer, toks)
            res[f"gen_text_{alpha}"] += gen_text

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)

    return df


def Ethos_tox(
    model_name: str, load_from_ckpt: bool, model=None, feat: int = 0
) -> pd.DataFrame:
    test = load_dataset(
        "iamollas/ethos",
        "binary",
        num_proc=4,
        trust_remote_code=True,
    )["train"]

    test = DataLoader(test, batch_size=32, num_workers=32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        model, _ = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "RuHae/model", token=HF_TOKEN, trust_remote_code=True
            ),
            transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", token=HF_TOKEN, padding_side="left"
            ),
        )
    else:
        model = model

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = None
        model.hook.mod_threshold = None
        model.hook.mod_scaling = None
        toks = tokenizer(sent["text"], return_tensors="pt", padding=True)

        gen_text = test_gen(model, tokenizer, toks)
        res["sent"] += sent["text"]
        res["sent_toxicity"] += sent["label"].tolist()

        res["gen_text_base"] += gen_text

        for alpha in [-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]:
            model.hook.mod_features = feat

            model.hook.mod_scaling = alpha

            gen_text = test_gen(model, tokenizer, toks)
            res[f"gen_text_{alpha}"] += gen_text

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)

    return df


def get_tox_existing(file: str):
    lock = Lock()
    text_cols = [
        "gen_text_base",
        "gen_text_0.0",
        "gen_text_-100.0",
        "gen_text_-50.0",
        "gen_text_-1.0",
        "gen_text_1.0",
        "gen_text_50.0",
        "gen_text_100.0",
        "gen_text_base",
        "gen_text_0.0",
        "gen_text_-100.0",
        "gen_text_-50.0",
        "gen_text_-1.0",
        "gen_text_1.0",
        "gen_text_50.0",
        "gen_text_100.0",
    ]
    new_cols = [
        "gen_toxicity_base",
        "gen_toxicity_0.0",
        "gen_toxicity_-100.0",
        "gen_toxicity_-50.0",
        "gen_toxicity_-1.0",
        "gen_toxicity_1.0",
        "gen_toxicity_50.0",
        "gen_toxicity_100.0",
        "gen_toxicity_base_total",
        "gen_toxicity_0.0_total",
        "gen_toxicity_-100.0_total",
        "gen_toxicity_-50.0_total",
        "gen_toxicity_-1.0_total",
        "gen_toxicity_1.0_total",
        "gen_toxicity_50.0_total",
        "gen_toxicity_100.0_total",
    ]
    data = pd.read_csv(
        file,
        index_col=False,
    )
    len_data = len(data)
    for new, text in zip(new_cols, text_cols):
        if new in data.columns:
            continue
        else:
            Process(
                target=create_col_new, args=(new, text, file, data, lock, len_data)
            ).start()


def create_col_new(
    new: str, text: str, file: str, df: pd.DataFrame, lock, len_data: int
) -> pd.DataFrame:
    with tqdm(total=len_data, desc=f"{file}: {text} -> {new}") as pbar:

        def get_tox(input: str) -> float:
            tox = tox_classifier(input)
            pbar.update(1)
            return tox

        if "total" in new:
            new_col = (df["sent"] + df[text]).apply(get_tox)
        else:
            new_col = df[text].apply(get_tox)

        lock.acquire()
        try:
            try:
                data = pd.read_csv(
                    file,
                    index_col=False,
                )
            except FileNotFoundError:
                data = df

            data[new] = new_col
            data.to_csv(
                file,
                index=False,
            )
        finally:
            lock.release()


def RTP_tox_feature(model, tokenizer, feat: int = 0, all: bool = False) -> None:
    test = load_from_disk("/nfs/scratch_2/ruben/SAE-FeatureExtraction/datasets/RTP")[
        "test"
    ]
    test = test.map(preprocess, num_proc=4)
    test = test.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    if not all:
        tmp = []
        for i in np.linspace(0.05, 1, 20):
            tmp.append(
                test.filter(
                    lambda x: x["text_toxicity"] >= i - 0.05 and x["text_toxicity"] < i
                )
                .shuffle(42)
                .select(range(100))
            )
        test = concatenate_datasets(tmp)
    test = DataLoader(test, batch_size=32, num_workers=16)

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = feat

        model.hook.mod_scaling = 1.0
        toks = tokenizer(
            sent["text"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        model(toks)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]

        res["sent"] += sent["text"]
        res["cont"] += sent["cont"]
        res["sent_toxicity"] += sent["text_toxicity"].tolist()
        res["cont_toxicity"] += sent["cont_toxicity"].tolist()
        res["tox_feature_value"] += latents[:, :, feat].cpu().tolist()
        res["latent_min"] += latents.min(2).values.cpu().tolist()
        res["latent_max"] += latents.max(2).values.cpu().tolist()
        res["latent_mean"] += latents.mean(2).cpu().tolist()
        res["latent_std"] += latents.std(2).cpu().tolist()
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)
    return df


def TC_tox_feature(model, tokenizer, feat: int = 0) -> None:
    test = load_dataset(
        "lmsys/toxic-chat",
        "toxicchat0124",
        split="test",
        num_proc=4,
    )
    test = DataLoader(test, batch_size=4, num_workers=16)

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = feat

        model.hook.mod_scaling = 1.0
        toks = tokenizer(
            sent["user_input"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        model(toks)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]

        res["sent"] += sent["user_input"]
        res["sent_toxicity"] += sent["toxicity"].tolist()
        res["tox_feature_value"] += latents[:, :, feat].cpu().tolist()
        res["latent_min"] += latents.min(2).values.cpu().tolist()
        res["latent_max"] += latents.max(2).values.cpu().tolist()
        res["latent_mean"] += latents.mean(2).cpu().tolist()
        res["latent_std"] += latents.std(2).cpu().tolist()
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)
    return df


def Ethos_tox_feature(model, tokenizer, feat: int = 0) -> None:
    test = load_dataset(
        "iamollas/ethos",
        "binary",
        num_proc=4,
        trust_remote_code=True,
    )["train"]
    test = DataLoader(test, batch_size=4, num_workers=16)

    res = defaultdict(list)
    for sent in tqdm(test):
        model.hook.mod_features = feat

        model.hook.mod_scaling = 1.0
        toks = tokenizer(
            sent["text"], return_tensors="pt", padding=True
        ).input_ids.cuda()
        model(toks)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]

        res["sent"] += sent["text"]
        res["sent_toxicity"] += sent["label"].tolist()
        res["tox_feature_value"] += latents[:, :, feat].cpu().tolist()
        res["latent_min"] += latents.min(2).values.cpu().tolist()
        res["latent_max"] += latents.max(2).values.cpu().tolist()
        res["latent_mean"] += latents.mean(2).cpu().tolist()
        res["latent_std"] += latents.std(2).cpu().tolist()
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(res)
    df = df.replace("\n", "", regex=True)
    return df


def RTP_tree(model, tokenizer, file, k, all: bool = True, model_name: str = "SCAR"):
    test = load_from_disk("/nfs/scratch_2/ruben/SAE-FeatureExtraction/datasets/RTP")[
        "test"
    ]
    if all:
        dataset_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] >= 0.5
        )
        dataset_non_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] < 0.5
        )
    else:
        dataset_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] >= 0.75
        )
        dataset_non_toxic = test.filter(
            lambda x: x["prompt"]["toxicity"] is not None
            and x["prompt"]["toxicity"] <= 0.0456735
        )

    def preprocess(example):
        is_toxic = example["prompt"]["toxicity"] >= 0.75
        return {
            "text": example["prompt"]["text"],
            "is_toxic": is_toxic,
            "toxcitity_level": example["prompt"]["toxicity"],
        }

    dataset_toxic = dataset_toxic.map(preprocess, num_proc=4)
    dataset_toxic = dataset_toxic.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    dataset_non_toxic = dataset_non_toxic.map(preprocess, num_proc=4)
    dataset_non_toxic = dataset_non_toxic.remove_columns(
        [
            "filename",
            "begin",
            "end",
            "challenging",
            "prompt",
            "continuation",
        ]
    )
    res = {"model": [], "depth": []}
    toxic_latents = []
    for line in tqdm(dataset_toxic):
        tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.cuda()
        model(tok_ids)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]
        topk = latents.topk(k).indices
        toxic_latents.append([tok_ids.detach(), latents.detach(), topk.detach()])

    non_toxic_latents = []
    for line in tqdm(dataset_non_toxic):
        tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.cuda()
        model(tok_ids)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]
        topk = latents.topk(k).indices
        non_toxic_latents.append([tok_ids.detach(), latents.detach(), topk.detach()])

    toxic_l_bin = []
    toxic_l_sum = []
    toxic_l_mean = []
    for ids, latents, inds in tqdm(toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    non_toxic_l_bin = []
    non_toxic_l_sum = []
    non_toxic_l_mean = []
    for ids, latents, inds in tqdm(non_toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        non_toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        non_toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        non_toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = toxic_l_mean + non_toxic_l_mean
    labels = [1 for _ in range(len(toxic_l_sum))] + [
        0 for _ in range(len(non_toxic_l_sum))
    ]
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf = clf.fit(ls, labels)
    print(clf.get_depth())
    res["model"].append(model_name)
    res["depth"].append(clf.get_depth())

    plt.figure(figsize=(40, 20))
    tree.plot_tree(
        clf,
        proportion=False,
        class_names=["non toxic", "toxic"],
        filled=True,
        max_depth=3,
    )
    plt.savefig(file)
    s = pickle.dumps(clf)
    with open(
        f"{file}.pkl",
        "wb",
    ) as f:
        f.write(s)

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{file}.csv")


def TC_tree(model, tokenizer, file, k, model_name: str = "SCAR"):
    test = load_dataset(
        "csv",
        data_files="/nfs/scratch_2/ruben/SAE-FeatureExtraction/SAE/model/SAE_eval/TC.csv",
    )["train"]
    dataset_toxic = test.filter(lambda x: x["toxicity_level"] >= 0.5)
    dataset_non_toxic = test.filter(lambda x: x["toxicity_level"] < 0.5)

    res = {"model": [], "depth": []}
    toxic_latents = []
    for line in tqdm(dataset_toxic):
        tok_ids = tokenizer(line["user_input"], return_tensors="pt").input_ids.cuda()
        model(tok_ids)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]
        topk = latents.topk(k).indices
        toxic_latents.append([tok_ids.detach(), latents.detach(), topk.detach()])

    non_toxic_latents = []
    for line in tqdm(dataset_non_toxic):
        tok_ids = tokenizer(line["user_input"], return_tensors="pt").input_ids.cuda()
        model(tok_ids)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]
        topk = latents.topk(k).indices
        non_toxic_latents.append([tok_ids.detach(), latents.detach(), topk.detach()])

    toxic_l_bin = []
    toxic_l_sum = []
    toxic_l_mean = []
    for ids, latents, inds in tqdm(toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    non_toxic_l_bin = []
    non_toxic_l_sum = []
    non_toxic_l_mean = []
    for ids, latents, inds in tqdm(non_toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        non_toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        non_toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        non_toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = toxic_l_mean + non_toxic_l_mean
    labels = [1 for _ in range(len(toxic_l_sum))] + [
        0 for _ in range(len(non_toxic_l_sum))
    ]
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf = clf.fit(ls, labels)
    print(clf.get_depth())
    res["model"].append(model_name)
    res["depth"].append(clf.get_depth())

    plt.figure(figsize=(40, 20))
    tree.plot_tree(
        clf,
        proportion=False,
        class_names=["non toxic", "toxic"],
        filled=True,
        max_depth=3,
    )
    plt.savefig(file)
    s = pickle.dumps(clf)
    with open(
        f"{file}.pkl",
        "wb",
    ) as f:
        f.write(s)

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{file}.csv")


def Ethos_tree(model, tokenizer, file, k, model_name: str = "SCAR"):
    test = load_dataset(
        "csv",
        data_files="/nfs/scratch_2/ruben/SAE-FeatureExtraction/SAE/model/SAE_eval/Ethos.csv",
    )["train"]
    dataset_toxic = test.filter(lambda x: x["toxicity_level"] >= 0.5)
    dataset_non_toxic = test.filter(lambda x: x["toxicity_level"] < 0.5)

    res = {"model": [], "depth": []}
    toxic_latents = []
    for line in tqdm(dataset_toxic):
        tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.cuda()
        model(tok_ids)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]
        topk = latents.topk(k).indices
        toxic_latents.append([tok_ids.detach(), latents.detach(), topk.detach()])

    non_toxic_latents = []
    for line in tqdm(dataset_non_toxic):
        tok_ids = tokenizer(line["text"], return_tensors="pt").input_ids.cuda()
        model(tok_ids)
        acts = model.hook.pop()
        latents = model.SAE(acts)[1]
        topk = latents.topk(k).indices
        non_toxic_latents.append([tok_ids.detach(), latents.detach(), topk.detach()])

    toxic_l_bin = []
    toxic_l_sum = []
    toxic_l_mean = []
    for ids, latents, inds in tqdm(toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    non_toxic_l_bin = []
    non_toxic_l_sum = []
    non_toxic_l_mean = []
    for ids, latents, inds in tqdm(non_toxic_latents):
        l_sum = latents.sum(1)
        l_mean = latents.mean(1)
        l_bin = (latents > 0).sum(1)
        non_toxic_l_bin.append(l_bin.detach().cpu().numpy()[0])
        non_toxic_l_sum.append(l_sum.detach().cpu().numpy()[0])
        non_toxic_l_mean.append(l_mean.detach().cpu().numpy()[0])

    ls = toxic_l_mean + non_toxic_l_mean
    labels = [1 for _ in range(len(toxic_l_sum))] + [
        0 for _ in range(len(non_toxic_l_sum))
    ]
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf = clf.fit(ls, labels)
    print(clf.get_depth())
    res["model"].append(model_name)
    res["depth"].append(clf.get_depth())

    plt.figure(figsize=(40, 20))
    tree.plot_tree(
        clf,
        proportion=False,
        class_names=["non toxic", "toxic"],
        filled=True,
        max_depth=3,
    )
    plt.savefig(file)
    s = pickle.dumps(clf)
    with open(
        f"{file}.pkl",
        "wb",
    ) as f:
        f.write(s)

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{file}.csv")
