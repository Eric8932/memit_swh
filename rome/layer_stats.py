import os
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

import json

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=False,
    progress=tqdm,
    force_recompute=False,
    c_noupt = False,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        if 'gpt-j' not in tokenizer.name_or_path:
            maxlen = model.config.max_position_embeddings
        else:
            maxlen = model.config.n_positions
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        if ds_name == 'wikipedia':
            raw_ds = load_dataset(
                ds_name,
                dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
                cache_dir='/apdcephfs/share_1157269/yirenchen/wenhangshi/data_tmp/wikipedia')#github
            # '/apdcephfs/share_1157269/yirenchen/wenhangshi/data_tmp/wikipedia'
            # /data/swh/UER/memit/resource/wiki

            return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
        elif ds_name == 'zsre_train':
            text_list = []
            with open('data/zsre_swh_train.json', "r") as f:
                raw = json.load(f)
            for i in range(len(raw)):
                text_list.append(raw[i]['src']+' '+raw[i]['answers'][0])
            data = {
            "id": list(range(1, 1+len(text_list))), 
            "text": text_list  # Your texts
            }

            r_data = Dataset.from_dict(data)
            return TokenizedDataset(r_data, tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if 'gpt-j' not in tokenizer.name_or_path:
        npos = model.config.max_position_embeddings
    else:
        npos = model.config.n_positions
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    if c_noupt and ('gpt-j' not in tokenizer.name_or_path or ds_name != 'wikipedia'):
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}_noupt.npz"
    else:
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if not filename.exists() and download:
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(#只计算二阶动量
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True#获取对应层的输入，并且这个层过完就停止了
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)#不断计算这个统计量
    return stat

def layer_stats_added(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
    old_stat = None,
    last_requests = None,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds_requests(requests):

        if 'gpt-j' not in tokenizer.name_or_path:
            maxlen = model.config.max_position_embeddings
        else:
            maxlen = model.config.n_positions
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        data = {
            "id": list(range(1, 1+len(requests))),  # IDs from 1 to 10
            "text": requests  # Your texts
            }

        r_data = Dataset.from_dict(data)
        return TokenizedDataset(r_data, tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = len(last_requests) if len(last_requests) <100 else 100  # Examine this many dataset texts at once
    if 'gpt-j' not in tokenizer.name_or_path:
        npos = model.config.max_position_embeddings
    else:
        npos = model.config.n_positions
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    # size_suffix = "" if sample_size is None else f"_{sample_size}"
    # if batch_tokens < npos:
    #     size_suffix = "_t{batch_tokens}" + size_suffix
    # if model_name is None:
    #     model_name = model.config._name_or_path.replace("/", "_")

    # stats_dir = Path(stats_dir)
    # file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    # filename = stats_dir / file_extension

    # assert filename.exists()

    ds = get_ds_requests(last_requests)

    if progress is None:
        progress = lambda x: x

    old_stat.cuda_()
    stat = old_stat
    loader = tally(#只计算二阶动量
        stat,
        ds,
        cache=None,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
        store=False,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True#获取对应层的输入，并且这个层过完就停止了
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)#不断计算这个统计量
    return stat


if __name__ == "__main__":
    main()

