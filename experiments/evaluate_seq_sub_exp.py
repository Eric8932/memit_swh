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

# 直接记录 src+para中能够做对的（分为一开始就能做对的，和编辑后能做对的）
# 第一个target token（如果第一个token预测对，后面token肯定也对），在最后一个token处，每层表征的经过映射后（有无LN），的排名和概率
# 两个模型，两个，都跑200条？
#场景不应该是连续编辑

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

    if max_tolerate_fail_num<10000:
        new_name += "_maxt"
        new_name += str(max_tolerate_fail_num)

    if not use_algo:
        new_name += "_zeroshot"
    if new_prompt:
        new_name += '_newprompt'
    if real_edit:
        new_name += '_real'


    new_name += '_seq'
    dir_name += '_sub_exp'
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

    case_result_template = str(run_dir / "{}_{}.json")



    afedit_res_list = []

    real_edit_num = 0
    last_records = None
    fail_seq_number = 0

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

    #1.先判断是否能够预测
    #1.1 如果能
    #1.看para能够预测的，把src+para都填上subject，得到src+para每条是否能够预测
    #2.再tok一下，得到最后一个位置的idx，然后得到每个样本每层last token的out表征
    #3.把上述所有表征（分为是否LN）经过最后一层映射
    #4.统计first target token（对于src和para一致），在vocab中的rank和prob

    #1.2 如果不能，也先跑一遍上述过程，然后编辑后再跑一遍
    save_dic = {}#key是case_id
    for record_chunks in chunks(ds, 1):
        record = record_chunks[0]

        record_dic = {}

        edit_ = edit_or_not(model,tok_loc,record)
        pre_success = []
        pre_success.append(not edit_)
        
        
        #得到一开始准确率和每层last token的
        #得到para的文本和准确率


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
        text_list = []
        text_list.append(record["requested_rewrite"]['prompt'].format(record["requested_rewrite"]['subject']))
        text_list += record["paraphrase_prompts"]
        target = record["requested_rewrite"]["target_new"]["str"]
        if ds_name == "mcf":
            target_true = record["requested_rewrite"]["target_true"]["str"]

        # idx_list = [len(a)-1 for a in tok_loc(text_list).input_ids]
        if 'gpt-j' not in tok.name_or_path:#llama的第一个token是1
            first_target_token = tok_loc(target).input_ids[1]
            if ds_name == 'mcf':
                first_true_token = tok_loc(target_true).input_ids[1]
        else:
            first_target_token = tok_loc(target).input_ids[0]
            if ds_name == 'mcf':
                first_true_token = tok_loc(target_true).input_ids[0]

        input_tok = tok_loc(text_list,return_tensors="pt",padding=True).to("cuda")

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
            layers=[
                hparams.layer_module_tmp.format(l)#loss_layer是最后一层，因为要得到最后一层的输出
                for l in range(hparams.v_loss_layer+1)
            ],
            retain_input=False,
            retain_output=True,#得到它的输出
        ) as tr:
            _ = model(**input_tok).logits

        #0是默认的，第二个索引是不同text，第三个索引是last token位置
        #每个文本，每层
        # print((ln_f(tr[hparams.layer_module_tmp.format(5)].output[0][0][idx_list[0]])@ lm_w + lm_b).shape)
        # print(torch.softmax(
        #     ln_f(tr[hparams.layer_module_tmp.format(5)].output[0][0][idx_list[0]])@ lm_w + lm_b
        #                            ,dim=-1).shape)
        
        full_repr_w_ln = [[torch.softmax(
            ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][-1])@ lm_w + lm_b
                                   ,dim=-1).detach().cpu()
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        
        full_repr_wo_ln = [[torch.softmax(
            tr[hparams.layer_module_tmp.format(l)].output[0][text_i][-1]@ lm_w + lm_b
                                   ,dim=-1).detach().cpu()
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        
        rank_prob_w_ln = []
        for i in range(len(full_repr_w_ln)):
            l_list = []
            for j in range(len(full_repr_w_ln[0])):
                prob = full_repr_w_ln[i][j]
                if ds_name == "mcf":
                    l_list.append((prob[first_target_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob[first_true_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_true_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
                                   
                else:
                    l_list.append((prob[first_target_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
            rank_prob_w_ln.append(l_list)

        rank_prob_wo_ln = []
        for i in range(len(full_repr_wo_ln)):
            l_list = []
            for j in range(len(full_repr_wo_ln[0])):
                prob = full_repr_wo_ln[i][j]
                if ds_name == "mcf":
                    l_list.append((prob[first_target_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob[first_true_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_true_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
                else:
                    l_list.append((prob[first_target_token].numpy(),int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
            rank_prob_wo_ln.append(l_list)

        record_dic['pre_rank_prob_w_ln'] = rank_prob_w_ln
        record_dic['pre_rank_prob_wo_ln'] = rank_prob_wo_ln

        # 不管是否成功，都继续编辑
        # if not edit_:
        #     save_dic[record["case_id"]] = record_dic
        #     continue

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
                
                #原来的3
                # for i in range(len(other_src_list)):
                #     X_text_list.append(other_src_list[i])
                #     y_list.append(1)
                #     words.append(record["requested_rewrite"]['subject'])
                #     X_text_list.append(other_src_list[i])
                #     y_list.append(-1)
                #     words.append(other_subject_list[i])
                # #正向前缀100条
                # for i in range(len(pos_prefix_list)):
                #     X_text_list.append(pos_prefix_list[i])
                #     y_list.append(1)
                #     words.append(record["requested_rewrite"]['subject'])
                #     X_text_list.append(random_prefix_list[i])
                #     y_list.append(-1)
                #     words.append(record["requested_rewrite"]['subject'])

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
            layers=[
                hparams.layer_module_tmp.format(l)#loss_layer是最后一层，因为要得到最后一层的输出
                for l in range(hparams.v_loss_layer+1)
            ],
            retain_input=False,
            retain_output=True,#得到它的输出
        ) as tr:
            _ = edited_model(**input_tok).logits

        #0是默认的，第二个索引是不同text，第三个索引是last token位置
        #每个文本，每层
        full_repr_w_ln = [[torch.softmax(
            ln_f(tr[hparams.layer_module_tmp.format(l)].output[0][text_i][-1])@ lm_w + lm_b
                                   ,dim=-1).detach().cpu()
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        
        full_repr_wo_ln = [[torch.softmax(
            tr[hparams.layer_module_tmp.format(l)].output[0][text_i][-1]@ lm_w + lm_b
                                   ,dim=-1).detach().cpu()
                     for l in range(hparams.v_loss_layer+1)] for text_i in range(len(text_list)) ]
        
        rank_prob_w_ln = []
        for i in range(len(full_repr_w_ln)):
            l_list = []
            for j in range(len(full_repr_w_ln[0])):
                prob = full_repr_w_ln[i][j]
                if ds_name == "mcf":
                    l_list.append((prob[first_target_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob[first_true_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_true_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
                else:
                    l_list.append((prob[first_target_token].numpy(),int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
            rank_prob_w_ln.append(l_list)

        rank_prob_wo_ln = []
        for i in range(len(full_repr_wo_ln)):
            l_list = []
            for j in range(len(full_repr_wo_ln[0])):
                prob = full_repr_wo_ln[i][j]
                if ds_name == "mcf":
                    l_list.append((prob[first_target_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob[first_true_token].numpy(),
                                   int(prob.sort(descending=True).indices.sort().indices[first_true_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
                else:
                    l_list.append((prob[first_target_token].numpy(),int(prob.sort(descending=True).indices.sort().indices[first_target_token].numpy()),
                                   prob.sort(descending=True).indices.numpy()[:10]))
            rank_prob_wo_ln.append(l_list)

        record_dic['after_rank_prob_w_ln'] = rank_prob_w_ln
        record_dic['after_rank_prob_wo_ln'] = rank_prob_wo_ln

        save_dic[record["case_id"]] = record_dic



        torch.cuda.empty_cache()
    # for k,v in record_dic.items():
    #     print(k)
    #     print(v)
    
    np.save(save_deltas_dir/("rank_prob2.npy"),save_dic)

        
                


    #保存不断累积的变化量（+）
    # np.save(save_deltas_dir/("deltas_seq.npy"),acc_delta)
    # #保存没被edit的record pass_record["case_id"]=pass_record["requested_rewrite"]
    # np.save(run_dir/("passrecord_"+str(num_edits)+".npy"),pass_record)
    # np.save(run_dir/("editedcord_"+str(num_edits)+".npy"),edited_record)
    # print(f"Results are saved in {run_dir}")


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
        z_diff = args.z_diff
    )
