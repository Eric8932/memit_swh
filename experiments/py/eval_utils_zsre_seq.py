"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.

prompt, rephrase, neighborhood都只有一条。但是会逐一拼接target的每个token，构成多个序列。然后评测时只关心最后一个位置的输出，返回Ture/False列表
评测是看预测的对不对，因为他本来就是事实。但是CF数据集只要概率更高就行了
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
from dsets import AttributeSnippets

def lower_and_strip_list(inp):
    return [i.lower().strip() for i in inp]

#判断是否要编辑--两个数据集通用
def edit_or_not(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
):
    subject, target_new = (#target_true不会用到，因为target_new就是target_true了吧？
        record["requested_rewrite"][x] for x in ["subject", "target_new"]
    )
    special_tokens = [tok.bos_token,tok.eos_token,tok.pad_token,tok.unk_token]
    target_new = target_new["str"].lower().strip()
    # target_new = normalize_text(target_new,special_tokens)
    
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]#只有一条
    input = tok(rewrite_prompts,return_tensors="pt").to('cuda')
    pred = tok.batch_decode(model.generate(
            input_ids=input["input_ids"], attention_mask=input["attention_mask"],
            num_beams=1, num_return_sequences=1, use_cache=True,max_new_tokens=15),
            skip_special_tokens=True,pad_token_id=tok.eos_token_id
            )[0][len(rewrite_prompts[0]):]
    
    
    # target_new = normalize_text(target_new.lower().strip(),special_tokens)
    # real_pred = normalize_text(pred.lower().strip(),special_tokens)[:len(target_new)]
    target_new = target_new.lower().strip()
    real_pred = pred.lower().strip()[:len(target_new)]

    if real_pred == target_new:
        return False
    else:
        return True

#zsre对于当前评测样本以及his的评测（没有loc）
def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    # snips: AttributeSnippets,
    # vec: TfidfVectorizer,
    model_name = None,
    model_path = None,
    new_prompt=False,#我在zsre数据集处理的时候就做了（zsre.py）
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ??? zsre数据集不需要计算fluency和consistency
    :param vec: ??? 
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (#target_true不会用到，因为target_new就是target_true了吧？
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]#只有一条
    paraphrase_prompts = record["paraphrase_prompts"]#只有一条
    neighborhood_prompts = record["neighborhood_prompts"]#不止一条，因为和每一个target_token拼接了
    loc_ans = record['loc_ans']#针对neiguborhood的target
    loc_prompt = record['loc_prompt']

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    use_llama = False
    if model_name in ['llama','vicuna']:
        use_llama = True
    if use_llama:
        #llama encode时，句首是1，这个1和其他id一起decode不会印象，但是单独解码会有影响。而且llama decode不会有空格的
        target_tok = tok(target_new["str"])["input_ids"][1:]#llama会单独识别句首的,原tokenizer decode单个词的时候都会带一个空格都会带一个空格，
    else:
        target_tok = tok(" " + target_new["str"])["input_ids"]#目标回答
    inp_prompts_og = list(chain(*prob_prompts))#所有的prompt形成一个list，这个任务就是1+1把
    #所有的prompt再和所有的target_token构成输入序列
    if use_llama:
        inp_prompts = [
            el + " "[:i]+ tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
    else:
        inp_prompts = [
            el + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets,use_llama)

    # Predict for neighborhood prompts (dictionary format).已经是构造成如上形式：prompt和target拼接，以及prompt
    # neighborhood_correct = test_batch_prediction_acc(
    #     model,
    #     tok,
    #     [
    #         el["prompt"].format(record["requested_rewrite"])#这个format是不是没啥用，因为没有{}给你放
    #         for el in neighborhood_prompts
    #     ],
    #     [el["target"] for el in neighborhood_prompts],#提前准备好的target的逐一拼接token的序列
    #     use_llama
    # )

    # probs = stuff_probs + neighborhood_correct
    probs = stuff_probs

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]#每个prompt他们的预测列表
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(#其实本质上就两个，因为prompt和rephrase在这个数据集里都只有一个
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    # ret["neighborhood_prompts_correct"] = neighborhood_correct

    #分别是prompt rephrase和neighborhhood，都是list形式的。前两个对应同一个target，loc有自己的target
    #直接generate，看生成是否包含target
    predin_prompt = [
        rewrite_prompts,
        paraphrase_prompts,
        loc_prompt
    ]
    inp_predin_prompt = list(chain(*predin_prompt))
    predin_ans = [target_new['str']]*len(rewrite_prompts)+[target_new['str']]*len(paraphrase_prompts)+[loc_ans]*len(loc_prompt)
    acc_l =generate_in_acc(model,inp_predin_prompt,predin_ans,model_name,model_path)
    ret["rewrite_predin"] = acc_l[:len(rewrite_prompts)]
    ret["rephrase_predin"] = acc_l[len(rewrite_prompts):len(rewrite_prompts)+len(paraphrase_prompts)]
    ret["loc_predin"] = acc_l[len(rewrite_prompts)+len(paraphrase_prompts):]

    return ret

#zsre之前评测loc
def compute_loc_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    model_name = None,
    model_path = None,
    new_prompt=False,#我在zsre数据集处理的时候就做了（zsre.py）
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ??? zsre数据集不需要计算fluency和consistency
    :param vec: ??? 
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.

    neighborhood_prompts = record["neighborhood_prompts"]#不止一条，因为和每一个target_token拼接了
    loc_ans = record['loc_ans']#针对neiguborhood的target
    loc_prompt = record['loc_prompt']

    # Flatten all the evaluated prefixes into one list.
    use_llama = False
    if model_name in ['llama','vicuna']:
        use_llama = True
   
    ret = {}
    # Predict for neighborhood prompts (dictionary format).已经是构造成如上形式：prompt和target拼接，以及prompt
    #其实不需要这个东西
    # neighborhood_correct = test_batch_prediction_acc(
    #     model,
    #     tok,
    #     [
    #         el["prompt"]
    #         for el in neighborhood_prompts
    #     ],
    #     [el["target"] for el in neighborhood_prompts],#提前准备好的target的逐一拼接token的序列
    #     use_llama
    # )

    
    # ret["neighborhood_prompts_correct"] = neighborhood_correct

    #分别是prompt rephrase和neighborhhood，都是list形式的。前两个对应同一个target，loc有自己的target
    #直接generate，看生成是否包含target
    predin_prompt = [
        loc_prompt
    ]
    inp_predin_prompt = list(chain(*predin_prompt))
    predin_ans = [loc_ans]*len(loc_prompt)
    assert len(inp_predin_prompt) == len(predin_ans)
    acc_l =generate_in_acc(model,inp_predin_prompt,predin_ans,model_name,model_path)
    ret["loc_predin"] = acc_l

    return ret


def normalize_text(s,special_tokens=[]):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):#不用
        regex = re.compile(r"\b(a|an|the|)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text,special_tokens):
        for spe in special_tokens:
            text = text.replace(spe,"")
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):#不用
        return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(s,special_tokens=special_tokens))

#用于之前评测当前样本+his+loc，generate来判断是否正确--直接输入prompts和targets就行（用于一条一条判断）
def generate_in_acc(model, prompts: typing.List[str], target,model_name,model_path):
    if model_name in ['llama','vicuna']:
        tok = LlamaTokenizer.from_pretrained(model_path,padding_side="left")
        tok.pad_token = '<unk>'
    else:
        tok = AutoTokenizer.from_pretrained(model_path,padding_side="left")
        tok.pad_token = tok.eos_token
    special_tokens = [tok.bos_token,tok.eos_token,tok.pad_token,tok.unk_token]
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
        ).to("cuda")
    # print(prompt_tok)
    pred = tok.batch_decode(model.generate(
            input_ids=prompt_tok["input_ids"], attention_mask=prompt_tok["attention_mask"],
                num_beams=1, num_return_sequences=1, use_cache=True,max_new_tokens=15),
                skip_special_tokens=True,pad_token_id=tok.eos_token_id
                )
    acc_l = []
    for i in range(len(pred)):
        predin, equl_acc = False, False
        real_pred = pred[i][len(prompts[i]):].lower().strip()
        temp_t = target[i].lower().strip()
        if temp_t in real_pred:#生成在里面
            predin = True
            
        # nor_t = normalize_text(temp_t,special_tokens)
        # nor_pred = normalize_text(real_pred,special_tokens)
        nor_t = temp_t
        nor_pred = real_pred
        if nor_t == nor_pred[:len(nor_t)]:
            equl_acc = True
        
        acc_l.append([predin,equl_acc])
    return acc_l



#zsre之前评测当前样本和his的时候（一条一条），计算旧指标（teacher-forcing）
def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target,use_llama):
    #长度一致，然后一起输入模型，
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
        ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits#得到所有位置最后的logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1#找到非pad部分的最后一个token
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)#
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)
        #target的每个token的id
        if not use_llama:
            correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
                "input_ids"
            ]
        else:
            correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ][:,1:]#llama开头会有一个bos
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()#虽然target token此时应该也只有一个

    return (ans == correct_id).detach().cpu().numpy().tolist()
            #看能预测对几个, T/F列表。只针对最后一个位置的预. 对于一个prompt+target，会有很多个预测？ 看看保存的是不是列表就知道了
    # else:#不用上面那个，因为我是left_pad，不能用attention_mask的加和找最后一个位置
    #     pred = model.generate(
    #         input_ids=prompt_tok["input_ids"], attention_mask=prompt_tok["attention_mask"],
    #             num_beams=1, num_return_sequences=1, use_cache=True,max_new_tokens=1,min_new_tokens=1)
    #     # src_len = prompt_tok["src_input_ids"].size(1)#原来的长度
    #     ans  = pred[:,-1].squeeze()
    #     correct_id = []
    #     temp_ids = tok(target, padding=True, return_tensors="pt").to("cuda")["input_ids"]
    #     temp_attn = tok(target, padding=True, return_tensors="pt").to("cuda")["attention_mask"]
    #     for i in range(len(temp_ids)):
    #         correct_id.append(temp_ids[i][temp_attn[i].index(1)])
    #     correct_id = torch.tensor(correct_id).to("cuda")
    # return (ans == correct_id).detach().cpu().numpy().tolist()


#zsre的新loc评测，传进来的tokenizer已经是left_padding
def zsre_loc_batch(model, tokenizer, data_loader,snips,vec):
    correct_count, total_count = 0, 0

    special_tokens = [tokenizer.bos_token,tokenizer.eos_token,tokenizer.pad_token,tokenizer.unk_token]
   
    with torch.no_grad():
        model.eval()
        model.to('cuda')
        for _, batch in enumerate(data_loader):
            trg = [b["trg"] for b in batch["raw"]]#以前的b["trg"]是列表
            input_ids = batch["src_input_ids"].to('cuda')
            attention_mask = batch["src_attention_mask"].to('cuda')
           
            ori_input = tokenizer.batch_decode(batch["src_input_ids"],skip_special_tokens=True)
           

            model_gen = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                min_length=0, num_beams=1, num_return_sequences=1, use_cache=True,max_new_tokens=20
            )
            pred = tokenizer.batch_decode(model_gen, skip_special_tokens=True)
                
            #1.截断
            for i in range(len(pred)):
                pred[i]=pred[i][len(ori_input[i]):]#+1是为了去掉空格，没有+1其实也无所谓

            acc = []
            #改成准确生成
            for t, p in zip(trg, pred):
                acc_or_not = False
                #标准化+取出target长度
                # t = normalize_text(t.lower().strip(),special_tokens)
                t = t.lower().strip()
                # if t ==  normalize_text(p.lower().strip(),special_tokens)[:len(t)]:
                if t == p.lower().strip()[:len(t)]:
                    acc_or_not= True
                acc.append(acc_or_not)
            acc = torch.tensor(acc).long()

            correct_count += torch.sum(acc)
            total_count += acc.size(0)

        return np.round(correct_count / total_count*100,2)
        


