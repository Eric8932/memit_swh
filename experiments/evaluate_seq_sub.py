import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM
from torch.utils.data import DataLoader

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
    ZSRE_Loc,
    MCF_Loc,
)
from experiments.py.eval_utils_counterfact_seq import compute_rewrite_quality_counterfact,mcf_loc_batch
from experiments.py.eval_utils_zsre_seq import compute_rewrite_quality_zsre,edit_or_not,zsre_loc_batch
from memit import MEMITHyperParams, apply_memit_to_model_sub
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *


ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model_sub),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "zsre": (MENDQADataset_Seq, compute_rewrite_quality_zsre,zsre_loc_batch),
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact,mcf_loc_batch),
   
}

#不要数据集了，直接弄成dataloader吧
LOC_DICT = {
    "zsre": ZSRE_Loc,
    "mcf": MCF_Loc,
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
    loc_batch_size = 4,
    orig_loc = True,
    final_loc = True,
    real_edit = False,
    c_noupt = False,
    c_adapt = False,
    neuron_num = 5,
    select_standard = "key_norm",
    M_from_sub = True,
    max_tolerate_fail_num = 10000,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]#超参数和应用的算法

    new_name = model_name.split('/')[-1]
    new_name += '_'
    new_name += ds_name
    new_name += '_'
    new_name += str(num_edits)

    new_name += '_sub'
    new_name += str(neuron_num)
    new_name += select_standard
    if not M_from_sub:
        new_name += 'Mall'

    if not use_algo:
        new_name += "_zeroshot"
    if new_prompt:
        new_name += '_newprompt'
    if real_edit:
        new_name += '_real'
    if c_noupt:
        new_name += '_cnoupt'
    if c_adapt:
        new_name += '_cadapt'


    new_name += '_seq'
    dir_name += '_sub'
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
            # model = LlamaForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16).cuda()
            model = LlamaForCausalLM.from_pretrained(model_path).cuda()#github
            tok = LlamaTokenizer.from_pretrained(model_path)
            tok.pad_token = '<unk>'#虽然它是单条tokenize以及评测，但是genearte时会一起tokenize，所以padding还是有用的
            print(f"vocab length={len(tok.get_vocab())}")
            
            #专门为loc弄的left_pad的tok--loc_dataloader和loc_eval
            tok_loc = LlamaTokenizer.from_pretrained(model_path,padding_side="left")
            tok_loc.pad_token = '<unk>'
        else:#models/gpt-j-6b
            # model = AutoModelForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16,).cuda()
            model = AutoModelForCausalLM.from_pretrained(model_path).cuda()#github
            tok = AutoTokenizer.from_pretrained(model_path)
            tok.pad_token = tok.eos_token#空的和<|endoftext|

            #专门为loc弄的left_pad的tok，其实后续的也可以直接用这个，没必要去分类模型类别了
            tok_loc = AutoTokenizer.from_pretrained(model_path,padding_side="left")
            tok_loc.pad_token = tok_loc.eos_token

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
    
    loc_data = LOC_DICT[ds_name](tok_loc,DATA_DIR,dataset_size=loc_data_size)
    loc_loader = DataLoader(loc_data, batch_size=loc_batch_size, collate_fn=loc_data.collate_fn)
    # ds_loc =LOC_DICT[ds_name](DATA_DIR, tok=tok, size=loc_data_size,llama=use_llama,new_prompt=new_prompt)

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

    case_result_template = str(run_dir / "{}_{}.json")

    if orig_loc:
        loc_start = time()
        if args.ds_name == 'zsre':
            equal_acc = ds_eval_loc(
                model,
                tok_loc,
                loc_loader,
                None,
                None
            )
            torch.cuda.empty_cache()
            loc_res = {}
            loc_res['equal_acc'] = equal_acc
        else:
            res = ds_eval_loc(
                model,
                tok_loc,
                loc_loader,
                snips,
                vec
            )
            torch.cuda.empty_cache()
            loc_res = {}
            loc_res['equal_acc'] = res[0]
            loc_res["ngram_entropy"] = res[1]
            loc_res['reference_score'] = res[2]
        loc_exec_time = time()-loc_start
        loc_res['loc_exec_time'] = loc_exec_time

        np.save(run_dir/("orig_loc.npy"),loc_res)

    afedit_res_list = []

    real_edit_num = 0
    last_records = None
    fail_seq_number = 0
    for record_chunks in chunks(ds, 1):#每一次都更新1个
        #更新前判断是否应该更新
        record = record_chunks[0]
        edit_ = edit_or_not(model,tok_loc,record)
        if not edit_:
            print("record has already success and pass")
            pass_record[record["case_id"]]=record["requested_rewrite"]
            continue

        real_edit_num+=1
        if real_edit and real_edit_num > num_edits:
                break
            
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
        edited_model,deltas,all_neuron_num = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            last_requests = last_records,
            c_noupt = c_noupt,
            neuron_num = neuron_num,
            select_standard = select_standard,
            M_from_sub = M_from_sub,
            **args_conserve_memory,
            **etc_args,#保存的事先计算好的kv对的地址，但是应该还没算出来
        )
        if c_adapt:
            last_records = [
                record["requested_rewrite"]["prompt"].format(record["requested_rewrite"]['subject']) +" "+ record["requested_rewrite"]["target_new"]["str"]
                for record in record_chunks
                ]
        exec_time = time() - start
        print("Execution took", exec_time)
        if acc_delta is None:
            acc_delta = {}
            with torch.no_grad():
                for w_name, (key_mat, val_mat,top_abs_indices) in deltas.items():
                    key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
                    upd_matrix = key_mat @ val_mat.T
                    new_matrix = torch.zeros(all_neuron_num, upd_matrix.shape[1], device='cuda')
                    new_matrix[top_abs_indices, :] = upd_matrix.float()
                    acc_delta[w_name] = new_matrix.float()
            
        else:
            with torch.no_grad():
                for w_name, (key_mat, val_mat,top_abs_indices) in deltas.items():
                    key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
                    upd_matrix = key_mat @ val_mat.T
                    new_matrix = torch.zeros(all_neuron_num, upd_matrix.shape[1], device='cuda')
                    new_matrix[top_abs_indices, :] = upd_matrix.float()
                    acc_delta[w_name] += new_matrix.float()

        # np.save(save_deltas_dir/"deltas_"+str(edit_num)+".npy",deltas)#不把每一个都保存了
        edit_record.append(record)
        edit_num+=1
        model = edited_model

        # Evaluate new model
        start = time()

        #一条一条评估,评估可以跳过
        print("{} record".format(edit_num))
        # if out_file.exists():
        #     print(f"Skipping {out_file}; already exists")
        #     continue
        
        metrics = {
            "case_id": record["case_id"],
            # "grouped_case_ids": case_ids,#一起修改的事实的id
            "num_edits": num_edits,
            "time": exec_time,
            "post": ds_eval_method(
                model,
                tok,
                record,
                # [None,None],
                model_name,
                model_path,
                new_prompt,
            ),
        }
        afedit_res_list.append(metrics)
        torch.cuda.empty_cache()

        if args.ds_name == 'zsre':
            if not metrics['post']['rewrite_predin'][0][1]:
                fail_seq_number +=1 
            else:
                fail_seq_number = 0
        else:
            if not metrics['post']['rewrite_prompts_correct'][0]:
                fail_seq_number +=1 
            else:
                fail_seq_number = 0

        #把编辑过的评估一遍
        if (edit_num+1)%eval_edited_freq==0:
            eval_res_list = []
            for record in edit_record:
                metrics = {
                    "case_id": record["case_id"],
                    "post": ds_eval_method(
                        model,
                        tok,
                        record,
                        # None,None,
                        model_name,
                        model_path,
                        new_prompt,
                    ),
                }
                eval_res_list.append(metrics)
                torch.cuda.empty_cache()
            out_file_eval = Path(case_result_template.format(num_edits,"eval_"+str(edit_num)))
            with open(out_file_eval, "w") as f:
                json.dump(eval_res_list, f, indent=1)

        if fail_seq_number > max_tolerate_fail_num:
            print("real_edit_num: ",real_edit_num)
            break
        #     loc_start= time()
        #     if args.ds_name == 'zsre':
        #         equal_list = []#每一个样本只有一个acc，也只有一个结果要看
        #         for record in ds_loc:
        #             res= ds_eval_loc(
        #                 model,
        #                 tok,
        #                 record,
        #                 *(
        #                     gen_test_vars
        #                     if record["case_id"] % generation_test_interval == 0
        #                     else [None, None]
        #                 ),  # Only test generation every generation_test_interval cases
        #                 model_name,
        #                 model_path,
        #                 new_prompt,
        #             )
        #             equal_list.append(res['loc_predin'][0][1])#只有一条，取0，只有True or false
        #             torch.cuda.empty_cache()
        #         loc_res = {}
        #         loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
        #     else:
        #         equal_list = []
        #         ngram_entropy_list=[]
        #         reference_score_list = []
        #         for record in ds_loc:
        #             res= ds_eval_loc(
        #                 model,
        #                 tok,
        #                 record,
        #                 *(
        #                     gen_test_vars
        #                     if record["case_id"] % generation_test_interval == 0
        #                     else [None, None]
        #                 ),  # Only test generation every generation_test_interval cases
        #                 model_name,
        #                 model_path,
        #                 new_prompt,
        #             )
        #             equal_list+=[r[1] for r in res["loc_predin_true"]]
        #             ngram_entropy_list.append(res["ngram_entropy"])
        #             reference_score_list.append(res["reference_score"])
        #             torch.cuda.empty_cache()
        #         loc_res = {}
        #         loc_res['equal_acc'] = np.round(equal_list.count(True)/len(equal_list)*100,2)
        #         loc_res["ngram_entropy"] = np.round(np.mean(ngram_entropy_list)*100,2)
        #         loc_res['reference_score'] = np.round(np.mean(reference_score_list)*100,2)
        #     loc_exec_time = time()-loc_start
        #     loc_res['loc_exec_time'] = loc_exec_time

        #     np.save(save_deltas_dir/("loc_"+str(edit_num)+".npy"),loc_res)

    #保存每天样本编辑后的结果的列表
    out_file_afedit = Path(case_result_template.format(num_edits,"per_afedit"))
    with open(out_file_afedit, "w") as f:
        json.dump(afedit_res_list, f, indent=1)

    #最后再全部跑一遍
    final_res_list = []
    for record in edit_record:
        metrics = {
            "case_id": record["case_id"],
            "post": ds_eval_method(
                model,
                tok,
                record,
                # [None,None],
                model_name,
                model_path,
                new_prompt,
            ),
        }
        final_res_list.append(metrics)
        torch.cuda.empty_cache()

    #保存最终的模型在所有edited_record上的表现
    out_file_final = Path(case_result_template.format(num_edits,'per_final'))
    with open(out_file_final, "w") as f:
        json.dump(final_res_list, f, indent=1)

    if final_loc:
        loc_start= time()
        if args.ds_name == 'zsre':
            equal_acc = ds_eval_loc(
                model,
                tok_loc,
                loc_loader,
                None,
                None
            )
            torch.cuda.empty_cache()
            loc_res = {}
            loc_res['equal_acc'] = equal_acc
        else:
            res = ds_eval_loc(
                model,
                tok_loc,
                loc_loader,
                snips, 
                vec
            )
            torch.cuda.empty_cache()
            loc_res = {}
            loc_res['equal_acc'] = res[0]
            loc_res["ngram_entropy"] = res[1]
            loc_res['reference_score'] = res[2]
        loc_exec_time = time()-loc_start
        loc_res['loc_exec_time'] = loc_exec_time
        
        np.save(run_dir/("final_loc.npy"),loc_res)


    
    #保存不断累积的变化量（+）
    np.save(save_deltas_dir/("deltas_seq.npy"),acc_delta)
    #保存没被edit的record pass_record["case_id"]=pass_record["requested_rewrite"]
    np.save(run_dir/("passrecord_"+str(num_edits)+".npy"),pass_record)
    np.save(run_dir/("editedcord_"+str(num_edits)+".npy"),edited_record)
    print(f"Results are saved in {run_dir}")


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
    parser.add_argument(
        "--loc_batch_size",
        type=int,
        default=32,
        help="Batch-Size for loc dataloader",
    )

    parser.add_argument(
        "--orig_loc",
        action="store_false",
        help="Whether compute loc before editing",
    )
    parser.add_argument(
        "--final_loc",
        action="store_false",
        help="Whether compute loc after editing",
    )

    parser.add_argument(
        "--real_edit",
        action="store_true",
        help="num_edits = the real editing sample number",
    )

    parser.add_argument(
        "--c_noupt",
        action="store_true",
        help="Using the calculated C in default, which is updated along the editing",#默认为false，使用不断更新的版本
    )
    parser.add_argument(
        "--c_adapt",
        action="store_true",
        help="adpat c to the edited requests",#默认为false，使用不断更新的版本
    )

    parser.add_argument(
        "--neuron_num",
        type=int,
        default=5,
        help="Number of selelcted sub neuron from W_out ",
    )
    parser.add_argument(
        "--select_standard",
        type=str,
        default="key_norm",
        help="Standard for selecting sub neurons--key_norm/key/random",
    )
    parser.add_argument(
        "--M_from_sub",
        action="store_false",
        help="Whether M is from sub neurons",#默认为True，也就是zs-cur_zs保持不变
    )

    parser.add_argument(
        "--max_tolerate_fail_num",
        type=int,
        default=10000,
        help="Exploring the max editing number, default to 0",
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
        loc_batch_size = args.loc_batch_size,
        orig_loc = args.orig_loc,
        final_loc = args.final_loc,
        real_edit= args.real_edit,
        c_noupt = args.c_noupt,
        c_adapt= args.c_adapt,
        neuron_num = args.neuron_num,
        select_standard = args.select_standard,
        M_from_sub = args.M_from_sub,
        max_tolerate_fail_num = args.max_tolerate_fail_num
    )
