import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MENDQADataset_Seq,
    MENDQADataset_Loc,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact_seq import compute_rewrite_quality_counterfact,compute_loc_counterfact
from experiments.py.eval_utils_zsre_seq import compute_rewrite_quality_zsre,edit_or_not,compute_loc_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact,compute_loc_counterfact),
    "zsre": (MENDQADataset_Seq, compute_rewrite_quality_zsre,compute_loc_zsre),
}
LOC_DICT = {
    "mcf": MultiCounterFactDataset,#数据集没有变化，不需要改其实
    "zsre": MENDQADataset_Loc,
}


#1.一条一条样本遍历 2.先判断是否是需要编辑的样本，否则不放进来 
#3.针对edited算吧，最后评测的时候，再把模型重新读取一下（loc在zsre里完全不相关，但是mcf里还是挺接近的）

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    use_algo: bool = True,
    # use_llama:bool = False,
    # use_vicuna:bool = False,
    new_prompt:bool = False,
    model_path = None,
    eval_edited_freq=50,
    loc_data_size=100,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]#超参数和应用的算法

    new_name = model_name.split('/')[-1]
    new_name += '_'
    new_name += ds_name
    new_name += '_'
    new_name += str(num_edits)
    if not use_algo:
        new_name += "_zeroshot"
    if new_prompt:
        new_name += '_newprompt'
    new_name += '_seq'

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / new_name / continue_from_run).exists()
    ):
        continue_from_run = None
    #文件夹要添加1.模型 2.样本几条 3.有无使用算法 4.有无new_prompt
    

    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name / new_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / new_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    save_deltas_dir = RESULTS_DIR / dir_name / new_name
    print(f"Results will be stored at {run_dir}")
    print(f"Model deltas be stored at {save_deltas_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")
    # Instantiate vanilla model
    use_llama = False
    if model_name in ['llama','vicuna']:
        use_llama = True
    if type(model_name) is str:
        print("Instantiating model")
        #llama tokenize一定是0开头，再加上所有的target都是" "开头的，因此要考虑是[1:]还是[2:]
        if model_name in ['llama','vicuna']:
        #/data/swh/UER/TencentPretrain/models/vicuna-7b',/data/swh/UER/TencentPretrain/models/llama/7b_new
            model = LlamaForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16).cuda()
            tok = LlamaTokenizer.from_pretrained(model_path)
            tok.pad_token = '<unk>'#虽然它是单条tokenize以及评测，但是genearte时会一起tokenize，所以padding还是有用的
            print(f"vocab length={len(tok.get_vocab())}")
            tok.name_or_path = tok.name_or_path+'llama'
        else:#models/gpt-j-6b
            model = AutoModelForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16,).cuda()
            tok = AutoTokenizer.from_pretrained(model_path)
            tok.pad_token = tok.eos_token#空的和<|endoftext|
        model.config._name_or_path = model_name
        print(model.config._name_or_path.replace("/", "_"))
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    print(model.config._name_or_path)
    print(tok.name_or_path)

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None#构建(r,t)对应的所有o
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None#利用上面wiki构建的实体关系计算tf-idf的矩阵

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method,ds_eval_loc = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit,llama=use_llama,new_prompt=new_prompt)#都会限制数据集的大小和编辑数量一致（尤其CF），因此只有一个chunk
    ds_loc =LOC_DICT[ds_name](DATA_DIR, tok=tok, size=loc_data_size,llama=use_llama,new_prompt=new_prompt)#取整个数据集

    #可以保留，因为针对每个样本计算cache
    cache_template = None
    if use_cache:
        if not new_prompt:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}_seq.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}_newprompt_seq.npz"
            )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    edit_record = []
    pass_record = {}
    edited_record = {}
    edit_num=0
    acc_delta = None

    case_result_template = str(run_dir / "{}_edits-case_{}_{}.json")
    gen_test_vars = [snips, vec]#计算fluency和consistency

    loc_start = time()
    if args.ds_name == 'zsre':
        equal_list = []#每一个样本只有一个acc，也只有一个结果要看
        for record in ds_loc:
            res= ds_eval_loc(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
                model_name,
                model_path,
                new_prompt,
            )
            equal_list.append(res['loc_predin'][0][1])#只有一条，取0，只有True or false
            torch.cuda.empty_cache()
        loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
    else:
        equal_list = []
        ngram_entropy_list=[]
        reference_score_list = []
        for record in ds_loc:
            res= ds_eval_loc(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
                model_name,
                model_path,
                new_prompt,
            )
            equal_list+=[r[1] for r in res["loc_predin_true"]]
            ngram_entropy_list.append(res["ngram_entropy"])
            reference_score_list.append(res["reference_score"])
            torch.cuda.empty_cache()
        loc_res = {}
        loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
        loc_res["ngram_entropy"] = np.round(np.mean(ngram_entropy_list)*100,2)
        loc_res['reference_score'] = np.round(np.mean(reference_score_list)*100,2)
    loc_exec_time = time()-loc_start
    loc_res['loc_exec_time'] = loc_exec_time

    np.save(save_deltas_dir/("orig_loc.npy"),loc_res)


    for record_chunks in chunks(ds, 1):#每一次都更新1个
        #更新前判断是否应该更新
        record = record_chunks[0]
        edit_ = edit_or_not(model,tok,record)
        if not edit_:
            print("record has already success and pass")
            pass_record[record["case_id"]]=record["requested_rewrite"]
            continue
        edited_record[record["case_id"]]=record["requested_rewrite"]

        # Compute weight changes + record weights that changed
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()
        #影响因素有model 数据集，new_prompt 编辑数量--可以存在run_dir里面，以后要评测可以直接读
        #直接跑算法，就重新跑一次吧
        edited_model, weights_copy,deltas = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,#保存的事先计算好的kv对的地址，但是应该还没算出来
        )
        exec_time = time() - start
        print("Execution took", exec_time)
        if acc_delta is None:
            acc_delta = {}
            with torch.no_grad():
                for w_name, (key_mat, val_mat) in deltas.items():
                    key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
                    upd_matrix = key_mat @ val_mat.T
                    acc_delta[w_name] = upd_matrix.float()
            
        else:
            with torch.no_grad():
                for w_name, (key_mat, val_mat) in deltas.items():
                    key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
                    upd_matrix = key_mat @ val_mat.T
                    acc_delta[w_name] += upd_matrix.float()

        # np.save(save_deltas_dir/"deltas_"+str(edit_num)+".npy",deltas)#不把每一个都保存了
        edit_record.append(record)
        edit_num+=1
        model = edited_model

        # Evaluate new model
        start = time()

        #一条一条评估,评估可以跳过
        print("{} record".format(edit_num))
        out_file = Path(case_result_template.format(num_edits, record["case_id"],"afedit"))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        
        metrics = {
            "case_id": record["case_id"],
            # "grouped_case_ids": case_ids,#一起修改的事实的id
            "num_edits": num_edits,
            "time": exec_time,
            "post": ds_eval_method(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
                model_name,
                model_path,
                new_prompt,
            ),
        }
        torch.cuda.empty_cache()

        with open(out_file, "w") as f:
            #保存每一条编辑后的模型在编辑样本上的表现
            json.dump(metrics, f, indent=1)
        print("Evaluation took", time() - start)

        #把编辑过的全部评估一遍
        if (edit_num+1)%eval_edited_freq==0:
            for record in edit_record:
                out_file = Path(case_result_template.format(num_edits, record["case_id"],"eval_"+str(edit_num)))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue
                #一条一条评估
                metrics = {
                    "case_id": record["case_id"],
                    "post": ds_eval_method(
                        model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                        model_name,
                        model_path,
                        new_prompt,
                    ),
                }
                torch.cuda.empty_cache()

                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)

            loc_start= time()
            if args.ds_name == 'zsre':
                equal_list = []#每一个样本只有一个acc，也只有一个结果要看
                for record in ds_loc:
                    res= ds_eval_loc(
                        model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                        model_name,
                        model_path,
                        new_prompt,
                    )
                    equal_list.append(res['loc_predin'][0][1])#只有一条，取0，只有True or false
                    torch.cuda.empty_cache()
                loc_res = {}
                loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
            else:
                equal_list = []
                ngram_entropy_list=[]
                reference_score_list = []
                for record in ds_loc:
                    res= ds_eval_loc(
                        model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                        model_name,
                        model_path,
                        new_prompt,
                    )
                    equal_list+=[r[1] for r in res["loc_predin_true"]]
                    ngram_entropy_list.append(res["ngram_entropy"])
                    reference_score_list.append(res["reference_score"])
                    torch.cuda.empty_cache()
                loc_res = {}
                loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
                loc_res["ngram_entropy"] = np.round(np.mean(ngram_entropy_list)*100,2)
                loc_res['reference_score'] = np.round(np.mean(reference_score_list)*100,2)
            loc_exec_time = time()-loc_start
            loc_res['loc_exec_time'] = loc_exec_time

            np.save(save_deltas_dir/("loc_"+str(edit_num)+".npy"),loc_res)


    #最后再全部跑一遍
    for record in edit_record:
        out_file = Path(case_result_template.format(num_edits, record["case_id"],'final'))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        #一条一条评估
        metrics = {
            "case_id": record["case_id"],
            "post": ds_eval_method(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
                model_name,
                model_path,
                new_prompt,
            ),
        }
        torch.cuda.empty_cache()

        # Dump metrics in .json
        #保存最终的模型在所有edited_record上的表现
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

    loc_start= time()
    if args.ds_name == 'zsre':
        equal_list = []#每一个样本只有一个acc，也只有一个结果要看
        for record in ds_loc:
            res= ds_eval_loc(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
                model_name,
                model_path,
                new_prompt,
            )
            equal_list.append(res['loc_predin'][0][1])#只有一条，取0，只有True or false
            torch.cuda.empty_cache()
        loc_res = {}
        loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
    else:
        equal_list = []
        ngram_entropy_list=[]
        reference_score_list = []
        for record in ds_loc:
            res= ds_eval_loc(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
                model_name,
                model_path,
                new_prompt,
            )
            equal_list+=[r[1] for r in res["loc_predin_true"]]
            ngram_entropy_list.append(res["ngram_entropy"])
            reference_score_list.append(res["reference_score"])
            torch.cuda.empty_cache()
        loc_res = {}
        loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
        loc_res["ngram_entropy"] = np.round(np.mean(ngram_entropy_list)*100,2)
        loc_res['reference_score'] = np.round(np.mean(reference_score_list)*100,2)
    loc_exec_time = time()-loc_start
    loc_res['loc_exec_time'] = loc_exec_time

    np.save(save_deltas_dir/("final_loc.npy"),loc_res)


    
    #保存不断累积的变化量（+）
    np.save(save_deltas_dir/("deltas_seq.npy"),acc_delta)
    #保存没被edit的record pass_record["case_id"]=pass_record["requested_rewrite"]
    np.save(save_deltas_dir/("passrecord_"+str(num_edits)+".npy"),pass_record)
    np.save(save_deltas_dir/("editedcord_"+str(num_edits)+".npy"),edited_record)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B","llama","vicuna"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--use_algo",
        action="store_false",
        help="directly evaluate without applying any algorithm",
    )
    # parser.add_argument(
    #     "--use_llama",
    #     action="store_true",
    #     help="change tokenize when using llama",
    # )
    # parser.add_argument(
    #     "--use_vicuna",
    #     action="store_true",
    #     help="change tokenize when using llama",
    # )
    parser.add_argument(
        "--new_prompt",
        action="store_true",
        help="change the prompt for src rephrase neighborhood for zsre dataset",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/swh/UER/TencentPretrain/models/vicuna-7b",
        help="Local model path",
        required=True,
    )
    parser.add_argument(
        "--eval_edited_freq",
        type=int,
        default=200,
        help="Frequency for evaluating all edited examples.",
    )
    parser.add_argument(
        "--loc_data_size",
        type=int,
        default=100,
        help="Size for loc data",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        use_algo = args.use_algo,
        # use_llama = args.use_llama,
        # use_vicuna = args.use_vicuna,
        new_prompt = args.new_prompt,
        model_path = args.model_path,
        eval_edited_freq = args.eval_edited_freq,
        loc_data_size = args.loc_data_size,
    )
