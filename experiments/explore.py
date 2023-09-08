import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LlamaForCausalLM
from torch.utils.data import DataLoader

from experiments.py.eval_utils_counterfact_seq import compute_rewrite_quality_counterfact,mcf_loc_batch
from experiments.py.eval_utils_zsre_seq import compute_rewrite_quality_zsre,edit_or_not,zsre_loc_batch
from memit import MEMITHyperParams, apply_memit_to_model_sub
from rome import ROMEHyperParams, apply_rome_to_model

from dsets import MENDQADataset_Seq,MultiCounterFactDataset

from util import nethook
from util.globals import *
from util.generate import generate_fast

from memit.compute_ks import compute_ks

# 两个模型，两个数据（记录能不能完成）
# 1.每层last subject token的key向量
# 2.key*neuron_norm后的向量
# 12看分布有没有规律
# 3.如果向量有些维度比较大，对应的neuron的vocab--记录每个neuron，映射到vacab后的top10token及对应概率（分为有无LN）

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model_sub),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
}

DS_DICT = {
    "zsre": (MENDQADataset_Seq),
    "mcf": (MultiCounterFactDataset),
   }

CONTEXT_TEMPLATES_CACHE = None

#1.读取数据集，2.每条数据判断能否 3.从底至上传播


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int = 1,
    new_prompt:bool = False,
    model_path = None,
    neuron_vocab = False#要不要每个neuron映射到vocab分布上的top 10 token及概率
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]#超参数和应用的算法

    new_name = ''
    if 'llama' in model_name:
        new_name += 'llama_'
    else:
        new_name += 'gptj_'
    new_name += ds_name
    new_name += '_'
    new_name += str(num_edits)

    dir_name = 'EXP'
  

    run_dir = RESULTS_DIR / dir_name / new_name 
    run_dir.mkdir(parents=True, exist_ok=True)

    save_deltas_dir = RESULTS_DIR / dir_name / new_name 
    print(f"Results will be stored at {run_dir}")
    # Get run hyperparameters
    params_path = HPARAMS_DIR / alg_name / hparams_fname
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
            # model = LlamaForCausalLM.from_pretrained(model_path).cuda()#github
            tok = LlamaTokenizer.from_pretrained(model_path)
            tok.pad_token = '<unk>'#虽然它是单条tokenize以及评测，但是genearte时会一起tokenize，所以padding还是有用的
            print(f"vocab length={len(tok.get_vocab())}")
            
            #专门为loc弄的left_pad的tok--loc_dataloader和loc_eval
            tok_loc = LlamaTokenizer.from_pretrained(model_path,padding_side="left")
            tok_loc.pad_token = '<unk>'
        else:#models/gpt-j-6b
            model = AutoModelForCausalLM.from_pretrained(model_path,revision="float16",torch_dtype=torch.float16,).cuda()
            # model = AutoModelForCausalLM.from_pretrained(model_path).cuda()#github
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

    model.eval()
    context_templates =[
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(#自己实现的快速并行生成的函数。针对下面5个prompt，每个生成1个长为10的序列
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You",
                     "We","Yes","But","They","And",
                     "Every","Any","Then","Firstly","Rarely",
                     "Meanwhile","Otherwise","Both","Perhaps","People",
                     "Even","Like","If","As","Since"],
                    n_gen_per_prompt=n_gen,#20个词，每个生成5句
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ] 
    contexts = [context for context_type in context_templates for context in context_type]
    f = open("data/genarated_prompts.txt","w",encoding='utf-8')
    for c in contexts:
        f.write(c)
        f.write("\n")
    f.close()



    # ds = DS_DICT[ds_name](DATA_DIR, tok=tok, size=dataset_size_limit,llama=use_llama,new_prompt=new_prompt)#都会限制数据集的大小和编辑数量一致（尤其CF），因此只有一个chunk
    
    # #1.读取数据集，2.每条数据判断能否 3.从底至上传播

    # # 1.每层last subject token的key向量
    # # 2.key*neuron_norm后的向量
    # # 3.如果向量有些维度比较大，对应的neuron的vocab--记录每个neuron，映射到vacab后的top10token及对应概率（分为有无LN）--这个不用每次都算

    # #每条record的id作为key，是否通过，以及对应每层的key和key*norm
    # records_dic = {}
    # neuron_vocab_dic = {}#这两个只算一次


    # for record_chunks in chunks(ds, 1):#每一次都更新1个
    #     #更新前判断是否应该更新
    #     record = record_chunks[0]
    #     edit_ = edit_or_not(model,tok_loc,record)
    #     r_dic = {}
    #     r_dic['pass'] = not edit_

    #     request = record["requested_rewrite"]
    #     if request["target_new"]["str"][0] != " ":
    #         # Space required for correct tokenization
    #         request["target_new"]["str"] = " " + request["target_new"]["str"]

    #     model_named_modules = {x[0]: x[1] for x in model.named_modules()}
        
    #     context_templates = get_context_templates(model, tok)

    #     with torch.no_grad():
    #         for layer in range(hparams.v_loss_layer+1):
    #             layer_ks = compute_ks(model, tok, [request], hparams, layer, context_templates).T 

    #             weight_name = f"{hparams.rewrite_module_tmp.format(layer)}"
    #             col_norms = torch.norm(model_named_modules[weight_name].weight, dim=0).unsqueeze(-1)#4096*11008
    #             target_key = col_norms * layer_ks

    #             r_dic["key_"+str(layer)] = layer_ks.cpu()
    #             r_dic["key_norm"+str(layer)] = target_key.cpu()
    #             del layer_ks, target_key,col_norms
    #             torch.cuda.empty_cache()

    #             if neuron_vocab:
    #                 if str(layer)+"_top20_values" in neuron_vocab_dic:
    #                     continue
    #                 w1 = model_named_modules[weight_name].weight
    #                 if 'llama' in model_name:
    #                     n = model_named_modules['model.norm']
    #                 else:
    #                     n = model_named_modules['transformer.ln_f']
    #                 lm = model_named_modules['lm_head']

    #                 v1 = lm(w1.t())
    #                 softmax_output = F.softmax(v1, dim=-1)
    #                 top20_values, top20_indices = torch.topk(softmax_output, 50, dim=-1)
    #                 # v1 = v1.cpu()
    #                 # softmax_output = softmax_output.cpu()
    #                 neuron_vocab_dic[str(layer)+"_top20_values"] = top20_values.cpu()
    #                 neuron_vocab_dic[str(layer)+"_top20_indices"] = top20_indices.cpu()
    #                 del v1,softmax_output,top20_values,top20_indices
    #                 torch.cuda.empty_cache()

    #                 w1 = n(w1.t())
    #                 v1 = lm(w1)
    #                 # w1 = w1.cpu() 
    #                 softmax_output_norm = F.softmax(v1, dim=-1)
    #                 # softmax_output_norm = softmax_output_norm.cpu()
    #                 top20_values_norm, top20_indices_norm = torch.topk(softmax_output_norm, 50, dim=-1)

    #                 neuron_vocab_dic[str(layer)+"_top20_values_norm"] = top20_values_norm.cpu()
    #                 neuron_vocab_dic[str(layer)+"_top20_indices_norm"] = top20_indices_norm.cpu()
    #                 del w1,v1,softmax_output_norm,top20_values_norm,top20_indices_norm
    #                 torch.cuda.empty_cache()

    #     records_dic[record['case_id']] = r_dic

    # np.save(run_dir/("key.npy"),records_dic)
    # if neuron_vocab:
    #     np.save(run_dir/("neuron_vocab_50.npy"),neuron_vocab_dic)



    # print(f"Results are saved in {run_dir}")


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(#自己实现的快速并行生成的函数。针对下面5个prompt，每个生成1个长为10的序列
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

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
        "--neuron_vocab",
        action="store_true",
        help="Whether compute loc before editing",
    )
    

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        num_edits=args.num_edits,
        new_prompt = args.new_prompt,
        model_path = args.model_path,
        neuron_vocab = args.neuron_vocab

    )
