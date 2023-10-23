import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
from copy import deepcopy

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

# 针对para，针对subject last和last token
# 两个target token，在最后一个token处，每层表征的经过映射后（有无LN），的logits prob rank
#不要其余信息了，然后每层的每个module的输出都要

def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"#这里给words前面加了空格.但是是深拷贝，所以不会影响原来的结果

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])
    if 'gpt-j' not in tok.name_or_path:#有bos
        prefixes_len, words_len, suffixes_len = [ 
            [len(el) for el in batch_tok.input_ids[i : i + n]] 
            for i in range(0, n * 3, n) 
            ]
    else:
        prefixes_tok, words_tok, suffixes_tok = [
            batch_tok[i : i + n] for i in range(0, n * 3, n)
        ]
        prefixes_len, words_len, suffixes_len = [
            [len(el) for el in tok_list]
            for tok_list in [prefixes_tok, words_tok, suffixes_tok]
        ]

    # Compute indices of last tokens
    if subtoken == "subject_last" :
        if 'gpt-j' not in tok.name_or_path:#两个开头都有1，但是因为这是用于索引，所以只去掉一个1.多去掉一个1因为words前面加了一个空格
            return [ [prefixes_len[i]+ words_len[i]-3 + (1 if prefixes_len[i] ==1 else 0)] for i in range(n) ]
        else:
            return [ [ prefixes_len[i]+ words_len[i] -1 ] for i in range(n) ]
    elif subtoken == "subject_first" :
        return [ [prefixes_len[i]] for i in range(n) ]
    elif subtoken == "subject_subseq":
        if 'gpt-j' not in tok.name_or_path:#两个开头都有1，但是因为这是用于索引，所以只去掉一个1.多去掉一个1因为words前面加了一个空格
            return [ [prefixes_len[i]+ words_len[i]-2 + (1 if prefixes_len[i] ==1 else 0)
                     - (1 if suffixes_len[i] == 1 else 0)] for i in range(n) ]
        else:
            return [ [ prefixes_len[i]+ words_len[i] - (1 if suffixes_len[i] == 0 else 0) ] for i in range(n) ]
        

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
    filter_standard = "1",
    pos_neg_construct = 1,
    M_from_sub = True,
    max_tolerate_fail_num = 10000,
    qr = False,
    z_diff = False,
    after_edit = True,
    top_number = 100,
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
    new_name +="_"
    new_name += select_standard
    new_name += str(pos_neg_construct)

    new_name += '_'
    new_name += hparams_fname.split(".")[0]

    if not use_algo:
        new_name += "_zeroshot"
    if new_prompt:
        new_name += '_newprompt'
    if real_edit:
        new_name += '_real'


    new_name += '_seq'
    dir_name += '_logits'
    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist

    save_deltas_dir = RESULTS_DIR / dir_name / new_name / "logits"
    save_deltas_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {save_deltas_dir}")

    # Get run hyperparameters
    params_path =  HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)

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

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method,ds_eval_loc = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit,llama=use_llama,new_prompt=new_prompt)#都会限制数据集的大小和编辑数量一致（尤其CF），因此只有一个chunk
    
    loc_data = LOC_DICT[ds_name](tok_loc,DATA_DIR,dataset_size=loc_data_size)

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
    edit_num=0


    real_edit_num = 0
    last_records = None

    #根据使用的方法读取数据集
    if select_standard in ["mean_dif","mi","f_stat","lr","svm","key_pos"]:
        #1.正向前缀
        pos_prefix_list = []
        with open(f"{DATA_DIR}/pos_prefix.txt",'r',encoding='utf-8') as f:
            for line in f.readlines():
                pos_prefix_list.append(line.strip())

        #2.其他src和对应subject
        other_src_subject_path = f"{DATA_DIR}/src_subject.txt"#正例，直接把subject填入src中
        other_src_list = []
        other_subject_list = []
        with open(other_src_subject_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                src,subject = line.strip().split('\t')
                other_src_list.append(src)
                other_subject_list.append(subject)

        #3.随机前缀
        random_prefix_list = []
        with open(f"{DATA_DIR}/random_prefix.txt",'r',encoding='utf-8') as f:
            for line in f.readlines():
                random_prefix_list.append(line.strip())

    
    save_dic = {}#key是case_id
    record_num = 0
    for record_chunks in chunks(ds, 1):
        record_num += 1
        # if record_num<=400:#前400条都做过了，不做了
        #     continue
        
        record = record_chunks[0]

        record_dic = {}

        edit_ = edit_or_not(model,tok_loc,record)
        pre_success = []
        pre_success.append(not edit_)
        

        metrics= ds_eval_method(
            model,
            tok,
            record,
            # [None,None],
            model_name,
            model_path,
            new_prompt)
 
        if ds_name == 'zsre':
            pre_success+=[metrics['rephrase_predin'][k][1] for k in range(len(metrics['rephrase_predin']))]#只有一条para
        else:
            pre_success += metrics['paraphrase_prompts_correct']
        record_dic['pre_success']=pre_success

        #构造文本列表
        context_templates = [record["requested_rewrite"]['prompt']]
        text_list = [record["requested_rewrite"]['prompt'].format(record["requested_rewrite"]['subject'])]
        words = [record["requested_rewrite"]['subject']]*len(text_list)

        subject_first_index = get_words_idxs_in_templates(tok,context_templates,words,"subject_first")


        subject_last_index = get_words_idxs_in_templates(tok,context_templates,words,"subject_last")
        last_index = [[-1]*len(text_list)]

        target_new = record["requested_rewrite"]["target_new"]["str"]
        if ds_name == "mcf":
            target_true = record["requested_rewrite"]["target_true"]["str"]

        # idx_list = [len(a)-1 for a in tok(text_list).input_ids]
        if 'gpt-j' not in tok.name_or_path:#llama的第一个token是1
            first_new_token = tok(target_new).input_ids[1]
            if ds_name == 'mcf':
                first_true_token = tok(target_true).input_ids[1]
        else:
            first_new_token = tok(" "+target_new).input_ids[0]
            if ds_name == 'mcf':
                first_true_token = tok(" "+target_true).input_ids[0]

        input_tok = tok(text_list,return_tensors="pt").to("cuda")

        lm_w, ln_f = (
            nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
            nethook.get_module(model, hparams.ln_f_module),
        )
        try:
            lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        except LookupError as _:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

        with nethook.TraceDict(
            module=model,
            layers=[ hparams.layer_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)] +
                   [ hparams.mlp_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)] +
                   [ hparams.attn_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)],
            retain_input=False,
            retain_output=True,#得到它的输出
        ) as tr:
            _ = model(**input_tok).logits

        layer_w_ln = [[((ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),

                            (ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu())
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        
        layer_wo_ln = [[((tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),

                            (tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                            (tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                            torch.softmax(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu())
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        #遍历所有文本，遍历每一层，每个module遍历是否ln，计算两个位置，分别对target_new/true的  logits prob rank top100 token
        def get_logits_prob_rank_top(rep_list,dataset):
            l = []
            l.append(rep_list[0][0][first_new_token].numpy())#layer_logits_subject_last_new
            l.append(rep_list[1][0][first_new_token].numpy())#layer_prob_subject_last_new
            l.append(int(rep_list[1][0].sort(descending=True).indices.sort().indices[first_new_token].numpy()))#layer_probe_rank_subject_last_new
            l.append(rep_list[1][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#layer_probe_top100_subject_last_new

            l.append(rep_list[2][0][first_new_token].numpy())#mlp_logits_subject_last_new
            l.append(rep_list[3][0][first_new_token].numpy())#mlp_prob_subject_last_new
            l.append(int(rep_list[3][0].sort(descending=True).indices.sort().indices[first_new_token].numpy()))#mlp_rank_subject_last_new
            l.append(rep_list[3][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#mlp_top100_subject_last_new

            l.append(rep_list[4][0][first_new_token].numpy())#attn_logits_subject_last_new
            l.append(rep_list[5][0][first_new_token].numpy())#attn_prob_subject_last_new
            l.append(int(rep_list[5][0].sort(descending=True).indices.sort().indices[first_new_token].numpy()))#attn_rank_subject_last_new
            l.append(rep_list[5][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#attn_top100_subject_last_new

            l.append(rep_list[6][0][first_new_token].numpy())#layer_logits_last_new
            l.append(rep_list[7][0][first_new_token].numpy())#layer_prob_last_new
            l.append(int(rep_list[7][0].sort(descending=True).indices.sort().indices[first_new_token].numpy()))#layer_logits_rank_last_new
            l.append(rep_list[7][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#layer_top100_last_new

            l.append(rep_list[8][0][first_new_token].numpy())#mlp_logits_last_new
            l.append(rep_list[9][0][first_new_token].numpy())#mlp_prob_last_new
            l.append(int(rep_list[9][0].sort(descending=True).indices.sort().indices[first_new_token].numpy()))#mlp_rank_last_new
            l.append(rep_list[9][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#mlp_top100_last_new

            l.append(rep_list[10][0][first_new_token].numpy())#attn_logits_last_new
            l.append(rep_list[11][0][first_new_token].numpy())#attn_prob_last_new
            l.append(int(rep_list[11][0].sort(descending=True).indices.sort().indices[first_new_token].numpy()))#attn_rank_last_new
            l.append(rep_list[11][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#attn_top100_last_new

            #true token
            if dataset == 'mcf':
                l.append(rep_list[0][0][first_true_token].numpy())#layer_logits_subject_last_true
                l.append(rep_list[1][0][first_true_token].numpy())#layer_prob_subject_last_true
                l.append(int(rep_list[1][0].sort(descending=True).indices.sort().indices[first_true_token].numpy()))#layer_rank_subject_last_true
                l.append(rep_list[1][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#layer_top100_subject_last_true
                
                l.append(rep_list[2][0][first_true_token].numpy())#mlp_logits_subject_last_true
                l.append(rep_list[3][0][first_true_token].numpy())#mlp_prob_subject_last_true
                l.append(int(rep_list[3][0].sort(descending=True).indices.sort().indices[first_true_token].numpy()))#mlp_rank_subject_last_true
                l.append(rep_list[3][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#mlp_top100_subject_last_true

                l.append(rep_list[4][0][first_true_token].numpy())#attn_logits_subject_last_true
                l.append(rep_list[5][0][first_true_token].numpy())#attn_prob_subject_last_true
                l.append(int(rep_list[5][0].sort(descending=True).indices.sort().indices[first_true_token].numpy()))#attn_rank_subject_last_true
                l.append(rep_list[5][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#attn_top100_subject_last_true
                
                l.append(rep_list[6][0][first_true_token].numpy())#layer_logits_last_true
                l.append(rep_list[7][0][first_true_token].numpy())#layer_prob_last_true
                l.append(int(rep_list[7][0].sort(descending=True).indices.sort().indices[first_true_token].numpy()))#layer_rank_last_true
                l.append(rep_list[7][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#layer_top100_last_true
                
                l.append(rep_list[8][0][first_true_token].numpy())#mlp_logits_last_true
                l.append(rep_list[9][0][first_true_token].numpy())#mlp_prob_last_true
                l.append(int(rep_list[9][0].sort(descending=True).indices.sort().indices[first_true_token].numpy()))#mlp_rank_last_true
                l.append(rep_list[9][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#mlp_top100_last_true

                l.append(rep_list[10][0][first_true_token].numpy())#attn_logits_last_true
                l.append(rep_list[11][0][first_true_token].numpy())#attn_prob_last_true
                l.append(int(rep_list[11][0].sort(descending=True).indices.sort().indices[first_true_token].numpy()))#attn_rank_last_true
                l.append(rep_list[11][0].sort(descending=True).indices.numpy()[:top_number].astype(np.int32))#attn_top100_last_true

            return [l]
        
        pre_w_ln = []
        for i in range(len(layer_w_ln)):#遍历所有文本
            l_list = []
            for j in range(len(layer_w_ln[0])):#遍历每一层
                rep = layer_w_ln[i][j]
                l_list += get_logits_prob_rank_top(rep,ds_name)
            pre_w_ln.append(l_list)
        record_dic['pre_w_ln'] = pre_w_ln
        del pre_w_ln

        pre_wo_ln = []
        for i in range(len(layer_wo_ln)):#遍历所有文本
            l_list = []
            for j in range(len(layer_wo_ln[0])):#遍历每一层
                rep = layer_wo_ln[i][j]
                l_list += get_logits_prob_rank_top(rep,ds_name)
            pre_wo_ln.append(l_list)
        record_dic['pre_wo_ln'] = pre_wo_ln
        del pre_wo_ln

        if after_edit:
            # Compute weight changes + record weights that changed
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()


            X_text_list = []
            y_list = []
            words = []
            if select_standard in ["mean_dif","mi","f_stat","lr","svm","key_pos"]:
                if pos_neg_construct == 1:
                    #正例-372条 
                    for i in range(len(other_src_list)):
                        X_text_list.append(other_src_list[i])
                        y_list.append(1)
                        words.append(record["requested_rewrite"]['subject'])
                    #负例-随机前缀+subject
                    for i in range(len(random_prefix_list)):
                        X_text_list.append(random_prefix_list[i])
                        y_list.append(-1)
                        words.append(record["requested_rewrite"]['subject'])
                    
                elif pos_neg_construct == 2:
                    #正例-372条 
                    #负例-其他subject
                    for i in range(len(other_src_list)):
                        X_text_list.append(other_src_list[i])
                        y_list.append(1)
                        words.append(record["requested_rewrite"]['subject'])
                        X_text_list.append(other_src_list[i])
                        y_list.append(-1)
                        words.append(other_subject_list[i])
                elif pos_neg_construct == 3:
                    #正向前缀300+其他src，负例：随机前缀300+其他src
                    for i in range(len(other_src_list)):
                        X_text_list.append(other_src_list[i])
                        y_list.append(1)
                        words.append(record["requested_rewrite"]['subject'])
                        X_text_list.append(other_src_list[i])
                        y_list.append(-1)
                        words.append(other_subject_list[i])
                    for j in range(3):
                        for i in range(len(pos_prefix_list)):
                            X_text_list.append(pos_prefix_list[i])
                            y_list.append(1)
                            words.append(record["requested_rewrite"]['subject'])
                            X_text_list.append(random_prefix_list[i+j*len(pos_prefix_list)])
                            y_list.append(-1)
                            words.append(record["requested_rewrite"]['subject'])
                elif pos_neg_construct == 4:
                    #正例：正向*3 负例：随机前缀
                    for j in range(3):
                        for i in range(len(pos_prefix_list)):
                            X_text_list.append(pos_prefix_list[i])
                            y_list.append(1)
                            words.append(record["requested_rewrite"]['subject'])
                            X_text_list.append(random_prefix_list[i+j*len(pos_prefix_list)])
                            y_list.append(-1)
                            words.append(record["requested_rewrite"]['subject'])
                        
                elif pos_neg_construct == 5:
                    #正例：正向*3 负例：其他src
                    for j in range(3):
                        for i in range(len(pos_prefix_list)):
                            X_text_list.append(pos_prefix_list[i])
                            y_list.append(1)
                            words.append(record["requested_rewrite"]['subject'])
                            X_text_list.append(other_src_list[i+j*len(pos_prefix_list)])
                            y_list.append(-1)
                            words.append(other_subject_list[i+j*len(pos_prefix_list)])
                            
                elif pos_neg_construct == 6:
                    #正例*2，负例：随机前缀+src
                    for i in range(len(other_src_list)):
                        X_text_list.append(other_src_list[i])
                        y_list.append(1)
                        words.append(record["requested_rewrite"]['subject'])
                        X_text_list.append(other_src_list[i])
                        y_list.append(-1)
                        words.append(other_subject_list[i])
                    for i in range(len(random_prefix_list)):
                        X_text_list.append(other_src_list[i])
                        y_list.append(1)
                        words.append(record["requested_rewrite"]['subject'])
                        X_text_list.append(random_prefix_list[i])
                        y_list.append(-1)
                        words.append(record["requested_rewrite"]['subject'])
                    
                y_list = np.array(y_list)

            edited_model,deltas,all_neuron_num,z = apply_algo(
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
                filter_standard = filter_standard,
                X_text_list = X_text_list,
                y_list = y_list,
                words = words,
                M_from_sub = M_from_sub,
                qr = qr,
                z_diff=z_diff,
                **args_conserve_memory,
                **etc_args,#保存的事先计算好的kv对的地址，但是应该还没算出来
            )
            if c_adapt:
                last_records = [
                    record["requested_rewrite"]["prompt"].format(record["requested_rewrite"]['subject']) +" "+ record["requested_rewrite"]["target_new"]["str"]
                    for record in record_chunks
                    ]

            edit_record.append(record)
            edit_num+=1
            # model = edited_model

            metrics= ds_eval_method(
                edited_model,
                tok,
                record,
                # [None,None],
                model_name,
                model_path,
                new_prompt)
            
            after_success = []

            if args.ds_name == 'zsre':
                after_success.append( metrics['rewrite_predin'][0][1])
                after_success+=[metrics['rephrase_predin'][k][1] for k in range(len(metrics['rephrase_predin']))]#只有一条para
            else:
                after_success.append(metrics['rewrite_prompts_correct'][0])
                after_success += metrics['paraphrase_prompts_correct']
            record_dic['after_success']= after_success

            lm_w, ln_f = (
                nethook.get_parameter(edited_model, f"{hparams.lm_head_module}.weight").T,
                nethook.get_module(edited_model, hparams.ln_f_module),
            )
            try:
                lm_b = nethook.get_parameter(edited_model, f"{hparams.lm_head_module}.bias")
            except LookupError as _:
                lm_b = next(edited_model.parameters()).new_zeros(edited_model.config.vocab_size)

        
            with nethook.TraceDict(
                module=edited_model,
                layers=[ hparams.layer_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)] +
                    [ hparams.mlp_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)] +
                    [ hparams.attn_module_tmp.format(l) for l in range(hparams.v_loss_layer+1)],
                retain_input=False,
                retain_output=True,
            ) as tr:
                _ = edited_model(**input_tok).logits


            layer_w_ln = [[((ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),

                                (ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(ln_f(tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(ln_f(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]])@ lm_w + lm_b,dim=-1).detach().cpu())
                        for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
            
            layer_wo_ln = [[((tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(tr[hparams.mlp_module_tmp.format(l)].output[text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][subject_last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),

                                (tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(tr[hparams.mlp_module_tmp.format(l)].output[text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu(),
                                (tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b).detach().cpu(),
                                torch.softmax(tr[hparams.attn_module_tmp.format(l)].output[0][text_i][last_index[text_i]]@ lm_w + lm_b,dim=-1).detach().cpu())
                        for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
            print(record["case_id"])
            after_w_ln = []
            for i in range(len(layer_w_ln)):#遍历所有文本
                l_list = []
                for j in range(len(layer_w_ln[0])):#遍历每一层
                    rep = layer_w_ln[i][j]
                    l_list += get_logits_prob_rank_top(rep,ds_name)
                after_w_ln.append(l_list)
            record_dic['after_w_ln'] = after_w_ln
            del after_w_ln

            after_wo_ln = []
            for i in range(len(layer_wo_ln)):#遍历所有文本
                l_list = []
                for j in range(len(layer_wo_ln[0])):#遍历每一层
                    rep = layer_wo_ln[i][j]
                    l_list += get_logits_prob_rank_top(rep,ds_name)
                after_wo_ln.append(l_list)
            record_dic['after_wo_ln'] = after_wo_ln
            del after_wo_ln
       

        save_dic[record["case_id"]] = record_dic

        if record_num % 100 == 0:
            np.save(save_deltas_dir/("logits2_"+str(record_num)+".npy"),save_dic)
            del save_dic
            save_dic = {}

        torch.cuda.empty_cache()
    # for k,v in record_dic.items():
    #     print(k)
    #     print(v)
    print(f"Results be stored at {save_deltas_dir}")
    np.save(save_deltas_dir/("logits2_final.npy"),save_dic)

        
                


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
        default="key",
        help="Standard for selecting sub neurons--key_norm/key/random",
    )
    parser.add_argument(
        "--filter_standard",
        type=str,
        default="1",
        help="Standard for filtering sub neurons--mean_dif/key_pos",
    )
    parser.add_argument(
        "--pos_neg_construct",
        type=int,
        default=1,
        help="Standard for selecting sub neurons--key_norm/key/random",
    )
    #1-正例-372条 负例-随机前缀+subject
    #2-正例-372条 负例-其他subject
    #3-正例-372  正向前缀+subject(100）  负例-#1随机前缀+subject(100条)+#2其他subject(372)

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

    parser.add_argument(
        "--qr",
        action="store_true",
        help="Using qr to solve the pseudo-inverse problem",#不用lstsq，因为它和伪逆的结果一致
    )

    parser.add_argument(
        "--z_diff",
        action="store_true",
        help="Saving z and cur_zs after editing, to explore the difference",#不用lstsq，因为它和伪逆的结果一致
    )

    parser.add_argument(
        "--after_edit",
        action="store_false",
        help="Saving z and cur_zs after editing, to explore the difference",#不用lstsq，因为它和伪逆的结果一致
    )
    parser.add_argument(
        "--top_number",
        type=int,
        default=100,
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
        filter_standard = args.filter_standard,
        pos_neg_construct = args.pos_neg_construct,
        M_from_sub = args.M_from_sub,
        max_tolerate_fail_num = args.max_tolerate_fail_num,
        qr = args.qr,
        z_diff = args.z_diff,
        after_edit=args.after_edit,
        top_number = args.top_number
    )
