import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from time import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats,layer_stats_added
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams
from rome import repr_tools

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import *

import random

#每一层不是更新整个矩阵，而是更新部分向量--维度数量1 5 10，选择的标准是key大小和key*norm（key是last subject token W_out矩阵输入）
#1.Z正常算，C正常算（不管是每层更新还是提前算好），但是C还是要adapt（后面把C换成用zsre_train去算）
#2.compute_ks得到每个requests（假设现在只有一个）的key，再得到对应的向量和norm，筛选出对应的维度

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_to_model_sub(#方法
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False, #zsre-False
    return_orig_weights=False, #zsre-True
    cache_template: Optional[str] = None,#存放kv对的地址。先计算kv对，才能计算
    last_requests = None,
    c_noupt = False,
    neuron_num = 5,
    select_standard = "key_norm",
    filter_standard = "1",
    X_text_list = None,
    y_list = None,
    words = None,
    M_from_sub = True,
    qr = False,
    z_diff = False,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    # weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas,z = execute_memit(model, tok, requests, hparams, cache_template=cache_template,last_requests=last_requests,c_noupt=c_noupt,neuron_num=neuron_num,
                           select_standard=select_standard,filter_standard=filter_standard,
                           X_text_list=X_text_list,y=y_list,words=words,M_from_sub=M_from_sub,qr=qr,z_diff=z_diff)#每一层的变化量（critical layers的W_out）


    all_neuron_num = None
    with torch.no_grad():
        for w_name, (key_mat, val_mat,top_abs_indices) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            if all_neuron_num is None:
                all_neuron_num = nethook.get_parameter(model, w_name).shape[1]
            # w = nethook.get_parameter(model, w_name)[:,top_abs_indices]#123
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w[:,top_abs_indices].shape)#123

            # if return_orig_weights and w_name not in weights_copy:
            #     weights_copy[w_name] = w.detach().clone()

            # w[...] += upd_matrix.float()#+=还是保持fp16，而且这种方法不会影响精度. #github
            # w[...] += upd_matrix.half()#+=还是保持fp16，而且这种方法不会影响精度.123

            w[:,top_abs_indices] += upd_matrix.float()#+=还是保持fp16，而且这种方法不会影响精度. #github
            # w[:,top_abs_indices] += upd_matrix.half()#+=还是保持fp16，而且这种方法不会影响精度.123

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model,deltas,all_neuron_num,z

#现在还只针对每次单条编辑
def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
    last_requests = None,
    c_noupt = False,#如果为True，就在更新前把所有层的C都算出来
    neuron_num = 5,
    select_standard = "key_norm",#key key_norm random
    filter_standard = "1",
    X_text_list = None,
    y = None,
    words = None,
    M_from_sub = True,#M为部分neuron的结果--直接减就行
    qr = False,
    z_diff = False,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)#requested_rewrite
    for i, request in enumerate(requests):
        #GPT-J/NEO的tokenizer要求target预留一个空格出来，只有compute_z里面用上了
        #为了得到target token
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:#打印前10个，看看prompt->target
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    #找到要改变的权重（对应层的MLP fc_out）
    print(hparams)
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers#critical layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)#prefix，用5个常见prompt开头迅速生成长为10的序列，以及一个{}
    z_layer = hparams.layers[-1]#目标层最后一层
    z_list = []

    #针对不同的request，计算出最后一层z_layer要输出的激活值，训练得到
    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        start = time()
        if not data_loaded:
            #得到目标层最后一层的值 z
            cur_z = compute_z(#通过梯度下降同时优化weight decay, kl_loss(1 prompt), nll_loss(5*1 prompt)，得到目标层的目标激活值
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
        exec_time = time() - start
        print("Execution took", exec_time)#10-7的时间，非常快，
    zs = torch.stack(z_list, dim=1)#构成矩阵的形式，不同request对应的值不同
    print("zs_shape",zs.shape)


    if c_noupt:#不随着层变化的更新而更新，在更新前提前算好所有的C。两个模型都用，因为还涉及c_adapt
        for i, layer in enumerate(hparams.layers):
            cov = get_cov(#随机采样输入计算原来W_out对应的K矩阵，用wikipedia采样10w条。而且针对当前层的fc_out的输入状态
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,#float32
                force_recompute=False,
                last_requests = last_requests,
                c_noupt = c_noupt,
            )
            cov.cpu()
            del cov


    #不同层的修改会叠加，导致每一次计算的layer_ks和targets其实都不同
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        #针对不同request，得到当前层的输入，堆叠成K矩阵（不同模版类型以及不同模版取平均）
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T 
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
        # print(layer_ks.shape)#11008*1

        #下面的逻辑只针对单条requests，也就是layer_ks.shape=n*1时
        if select_standard =='key':
            #找到值最大的5个位置
            top_indices = torch.argsort(layer_ks, dim=0, descending=True)[:neuron_num].flatten()
            # print("最大值的5个位置：", top_indices)

            # 找到绝对值最大的5个位置
            top_abs_indices = torch.argsort(layer_ks.abs(), dim=0, descending=True)[:neuron_num].flatten()
            print("绝对值最大的位置：", top_abs_indices)
            
        elif select_standard == 'key_norm':
            #得到每个W_out矩阵对应的norm
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"#4096*11008
            col_norms = torch.norm(weights[weight_name], dim=0).unsqueeze(-1)
            target_key = col_norms * layer_ks

            top_indices = torch.argsort(target_key, dim=0, descending=True)[:neuron_num].flatten()
            # print("最大值的5个位置：", top_indices)

            # 找到绝对值最大的5个位置
            top_abs_indices = torch.argsort(target_key.abs(), dim=0, descending=True)[:neuron_num].flatten()
            print("绝对值最大的位置：", top_abs_indices)
        elif select_standard == 'random':
            top_abs_indices = torch.randperm(layer_ks.size(0))[:neuron_num].flatten()
            print("随机选择的位置：", top_abs_indices)

        #275365非常慢，这个明天再全部跑一遍.太慢并且效果还不好，可以考虑不要
        #MI非常慢，但也要考虑
        elif select_standard in ["mean_dif","mi","f_stat","lr","svm","key_pos"]:
            #构造数据集应该提前做，然后每层再来计算
            
            #该函数得到每个样本在last subject token处的表征.现在有X_text_list--X是每个样本对应的向量
            X = repr_tools.get_reprs_at_word_tokens(model,tok,X_text_list,words,layer,module_template=hparams.rewrite_module_tmp,subtoken="last",track= "in")
            X = X.reshape(len(y),-1).detach().cpu().numpy()
            #构造X和y，其中X是last subject token的表征，y取决于是否是正的向量，然后转成numpy格式

            def get_heuristic_neuron_ranking(X, y, method,neuron_num):
                pos_class = y == 1
                if method == 'lr':
                    lr = LogisticRegression(
                        class_weight='balanced', C=0.1,
                        penalty='l1', solver='saga', n_jobs=-1)
                    lr = lr.fit(X, y)
                    ranks = np.argsort(np.abs(lr.coef_[0]))
                elif method == 'svm':
                    svm = LinearSVC(loss='hinge')
                    svm = svm.fit(X, y)
                    ranks = np.argsort(np.abs(svm.coef_))
                elif method == 'f_stat':
                    f_stat, p_val = f_classif(X, y)
                    ranks = np.argsort(f_stat)
                elif method == 'mi':
                    mi = mutual_info_classif(X, y)
                    ranks = np.argsort(mi)

                #abs合理嘛--abs找出差异最大的地方。
                #但我不仅要差异大，我要的就是值比你大--你不激活的地方我激活
                elif method == 'mean_dif':
                    mean_dif = np.abs(X[pos_class].mean(axis=0) -
                                    X[~pos_class].mean(axis=0))
                    # mean_dif = X[pos_class].mean(axis=0) -X[~pos_class].mean(axis=0)
                    ranks = np.argsort(mean_dif)
                elif method == 'key_pos':
                    pos_key = X[pos_class].mean(axis=0)
                    ranks = np.argsort(pos_key)
                else:
                    raise ValueError('Invalid method')
                ranks = ranks.flatten()
                return ranks[-neuron_num:]
            
            pos_class = y == 1
            if filter_standard =="mean_dif":
                mean_dif = np.abs(X[pos_class].mean(axis=0) -
                                X[~pos_class].mean(axis=0))
                # mean_dif = X[pos_class].mean(axis=0) -X[~pos_class].mean(axis=0)
                ranks = np.argsort(mean_dif)
                filter_size = 2000 if neuron_num <2000 else 20000
                filtered_subset = np.sort(ranks[-filter_size:])
                X_filtered = X[:, filtered_subset]
                top_filtered_ranks = get_heuristic_neuron_ranking(X_filtered, y, select_standard,neuron_num)

                top_abs_indices = torch.tensor(filtered_subset[top_filtered_ranks])
            elif filter_standard == "key_pos":
                pos_key = X[pos_class].mean(axis=0)
                ranks = np.argsort(pos_key)
                filter_size = 2000 if neuron_num <2000 else 20000
                filtered_subset = np.sort(ranks[-filter_size:])
                X_filtered = X[:, filtered_subset]
                top_filtered_ranks = get_heuristic_neuron_ranking(X_filtered, y, select_standard,neuron_num)

                top_abs_indices = torch.tensor(filtered_subset[top_filtered_ranks])
            else:
                top_abs_indices = torch.tensor(get_heuristic_neuron_ranking(X, y, select_standard,neuron_num))
            
        top_abs_indices = top_abs_indices.sort().values
        # print(top_abs_indices)
        
        #得到了top5_abs_indices

        # Compute residual error
        #得到不同request在z_layer（最后一层）的subject最后一个token的激活状态--输出
        #反正你也只要out，那还不如直接写激活值
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        # print("zs.shape ",zs.shape)#4096*1
        # print("cur_zs.shape ",cur_zs.shape)#4096*1
        targets = zs - cur_zs#hidden_size*修改个数

        #这个点先不考虑
        # if not M_from_sub:#假设M'来自全部neuron，那这里需要加上不更新neuron的输出
        #     #layer_ks.shape--11008*1
        #     weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"#4096*11008
        #     w = weights[weight_name]
        #     #把layer_ks中top_abs_indices置0，
        #     masked_layer_ks = layer_ks.clone()
        #     masked_layer_ks[top_abs_indices, :] = 0

        #     # 再和weight相乘,维度变成和targets一样
        #     result = torch.mm(w, masked_layer_ks).reshape(targets.shape)
        #     #，再相加
        #     targets += result

        #     del masked_layer_ks


        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))#应该是1，都是修改的个数
        targets = targets.repeat_interleave(repeat_factor, dim=1)#重复到修改个数

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]#只有第一层才算这个，后面的每层都应该用第一层算好的了

        #每个样本在当前层（fc_out）的，每个非pad/mask的token的输入状态的二阶动量
        #虽然会保存，但是保存的也是更新后的版本
        cov = get_cov(#随机采样输入计算原来W_out对应的K矩阵，用wikipedia采样10w条。而且针对当前层的fc_out的输入状态
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,#float32
            force_recompute=force_recompute,
            last_requests = last_requests,
            c_noupt = c_noupt,
        )
        
        #cov的维度应该就是11008*11008，用top5_abs_indices筛选出来新的cov和layer_ks
        # cov = cov[top_abs_indices, :][:, top_abs_indices].cpu()

        cov = cov[top_abs_indices, :].double().cpu()#123
        # cov = cov.double().cpu()#123

        sub_layer_ks = layer_ks[top_abs_indices].double().cpu()#123

        # Compute update in double precision
        # layer_ks, targets = (
        #     layer_ks.double(),
        #     targets.double(),
        # )
        #在这一步GPU不够
        layer_ks, targets = (
            layer_ks.double().cpu(),
            targets.double().cpu(),
        )
        

        # adj_k = torch.linalg.solve(
        #     hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
        #     layer_ks,
        # )

        if not qr:
            #算伪逆太慢了,而且最小二乘的结果估计是一样的
            solution = torch.linalg.lstsq((hparams.mom2_update_weight*cov + sub_layer_ks @ layer_ks.T).T,layer_ks)
            adj_k = solution[0]

            # pseudo_inv = torch.linalg.pinv(hparams.mom2_update_weight*cov + sub_layer_ks @ layer_ks.T)
            # adj_k = (layer_ks.T @ pseudo_inv).T
        else:
            Q, R = torch.linalg.qr((hparams.mom2_update_weight*cov + sub_layer_ks @ layer_ks.T).T)
            y = Q.T @ layer_ks
            adj_k = torch.linalg.solve(R, y)[:cov.size(0), :]

        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers 还有几层可以分散

        upd_matrix = (resid @ adj_k.T).cuda()


        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name][:,top_abs_indices].shape)

        # print("orig norm", torch.linalg.norm(weights[weight_name]))#cuda0 float 16
        # print("upd norm", torch.linalg.norm(upd_matrix))#cuda0 float64

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            #这样写没问题，并且还稍微提升了一点表现，所以最终的表现差异应该是来自于fp16和fp32的转换
            weights[weight_name][:,top_abs_indices] = weights_copy[weight_name][:,top_abs_indices] + upd_matrix.float()#github
            # weights[weight_name][:,top_abs_indices] = weights_copy[weight_name][:,top_abs_indices] + upd_matrix.half()#123变成了fp32，只影响后续的计算
            # weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.half()#123
            # print(weights_copy[weight_name].device,weights_copy[weight_name].dtype)
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
                top_abs_indices.cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    if z_diff:
        cache_z = {}
        cache_z['zs'] = zs.cpu()
        cur_v = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,#这里之前只得到RS，因此直接输入一整层。但我要得到W_out的输入
            fact_token_strategy=hparams.fact_token,
        )[1].T.cpu()
        #这里得到层输出

        cur_v = cur_v.cpu()
        cache_z['cur_zs'] = cur_v#1*11008

        #得到MLP层的输入作为hidden_state  
        hs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.mlp_module_tmp,#这里之前只得到RS，因此直接输入一整层。但我要得到W_out的输入
            fact_token_strategy=hparams.fact_token,
        )[0].T.cpu()

        #得到W_out层的输入，输入经过W_out再加上HS作为该层的输出 
        cur_k = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.rewrite_module_tmp,#这里之前只得到RS，因此直接输入一整层。但我要得到W_out的输入
            fact_token_strategy=hparams.fact_token,
        )[0]   

        #每一层都有重要的neuron，但是因为最后一层刚好就直接输出了，只选取最后一层的
        cur_k[:,top_abs_indices] *=1.1


        if 'gpt-j' not in tok.name_or_path:
            target_l = model.model.layers[10].mlp.down_proj
        else:
            target_l = model.transformer.h[8].mlp.fc_out
        new_v = target_l(cur_k).cpu()

        cache_z['new_zs'] = new_v+hs
        
    else:
        cache_z = None
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")


    return deltas ,cache_z


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,#默认就是false
    force_recompute: bool = False,
    last_requests = None,
    c_noupt = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if last_requests is None:
        if key not in COV_CACHE or force_recompute:
            stat = layer_stats(##GPTJ默认使用给定的C，只有LLaMA且c_noupt=True才会
                model,
                tok,
                layer_name,
                STATS_DIR,
                mom2_dataset,
                to_collect=["mom2"],
                sample_size=mom2_n_samples,
                precision=mom2_dtype,
                force_recompute=force_recompute,
                c_noupt = c_noupt,
            )
            # COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
            stat.cpu_()
            COV_CACHE[key] = stat
    else:
        stat = layer_stats_added(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=None,
            precision=mom2_dtype,
            old_stat=COV_CACHE[key],
            last_requests=last_requests,
        )
        stat.cpu_()
        COV_CACHE[key] = stat
    return (
        torch.inverse(COV_CACHE[key].mom2.moment().float().to("cuda")) if inv else COV_CACHE[key].mom2.moment().float().to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


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
