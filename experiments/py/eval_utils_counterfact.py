"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.

和zsre的区别在于 1.除了target_true 还要关心target_new，并且对于prompt, rephrase, neighbor目标不同 2.除了预测正确，还看预测概率是谁大，
3.还用tf-idf算一致性，n-gram_entropy算consistency 4.每个target只有一个概率（累加平均）
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    model_name = None,
    model_path = None,
    new_prompt=False,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    #对于一个事实来说，prompt, rephrase, neighborhood的target new/true都是一样的。
    #我补充一个，分别计算 target_new和target_true落在
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]#一条，并且挖出了subject
    paraphrase_prompts = record["paraphrase_prompts"]#两条
    neighborhood_prompts = record["neighborhood_prompts"]#多条
    generation_prompts = record["generation_prompts"]#多条

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    #前两个是要预测错，最后一个要预测对（其实不是对错，而是概率高低）--0对应new 1对应true
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    #输入的都是原prompt，还没拼接
    use_llama = False
    if model_name in ['llama','vicuna']:
        use_llama = True
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),#flatten了
        target_new["str"],
        target_true["str"],
        use_llama,
    )#返回prompt分别拼接target_new/true的预测概率--字典列表，以及预测正确对应的列表

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()#np.cumsum(1,2,n)
    #再重新按照prompt, rephrase, ntighbor拼成一个列表
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } 
    ret2 = {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }
    ret.update(ret2)

    inp_predin_prompt = list(chain(*prob_prompts))
    predin_ans_true = [target_true['str']]*len(rewrite_prompts)+[target_true['str']]*len(paraphrase_prompts)+[target_true['str']]*len(neighborhood_prompts)
    predin_ans_new = [target_new['str']]*len(rewrite_prompts)+[target_new['str']]*len(paraphrase_prompts)+[target_new['str']]*len(neighborhood_prompts)
    acc_l_true,acc_l_new =generate_in_acc(model,inp_predin_prompt,predin_ans_true,predin_ans_new,model_name,model_path)
    ret["rewrite_predin_true"] = acc_l_true[:len(rewrite_prompts)]
    ret["rephrase_predin_true"] = acc_l_true[len(rewrite_prompts):len(rewrite_prompts)+len(paraphrase_prompts)]
    ret["loc_predin_true"] = acc_l_true[len(rewrite_prompts)+len(paraphrase_prompts):]
    ret["rewrite_predin_new"] = acc_l_new[:len(rewrite_prompts)]
    ret["rephrase_predin_new"] = acc_l_new[len(rewrite_prompts):len(rewrite_prompts)+len(paraphrase_prompts)]
    ret["loc_predin_new"] = acc_l_new[len(rewrite_prompts)+len(paraphrase_prompts):]

    #不会每条事实都去算，默认是1
    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]#获取所有的r,o一样的wiki样本-o针对target_new
        #r,o 和subject都要一样的文本
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,#只是用于生成的prompt
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

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
        # exclude = set(string.punctuation)
        exclude = set(("?",":"))
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):#不用
        return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(s,special_tokens=special_tokens))

def generate_in_acc(model, prompts: typing.List[str], target_true, target_new, model_name,model_path):
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
                skip_special_tokens=True
                )
    acc_l_true = []
    acc_l_new = []
    for i in range(len(pred)):
        predin_true, equl_acc_true = False, False
        predin_new, equl_acc_new = False, False
        real_pred = pred[i][len(prompts[i]):].lower().strip()#就算有空格，也被去掉了，如果不是空格，就更不应该去掉了
        temp_target_true = target_true[i].lower().strip()
        temp_target_new = target_new[i].lower().strip()
        if temp_target_true in real_pred:
            predin_true = True
        if temp_target_new in real_pred:
            predin_new = True

        nor_true = normalize_text(temp_target_true,special_tokens)
        nor_new = normalize_text(temp_target_new,special_tokens)
        nor_pred = normalize_text(real_pred,special_tokens)
        if nor_true == nor_pred[:len(nor_true)]:
            equl_acc_true = True
        if nor_new == nor_pred[:len(nor_new)]:
            equl_acc_new = True
        acc_l_true.append([predin_true,equl_acc_true])
        acc_l_new.append([predin_new,equl_acc_new])
    return acc_l_true,acc_l_new

#分别拼接target_new/true后，模型对于target的预测概率（累加取平均）的大小
def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
    use_llama,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]#每个输入tokenized的长度
    #分别拼接两个target成一个完整的句子，再tokenize

    prompt_tok = tok(
    [
        f"{prefix} {suffix}"
        for prefix in prefixes
        for suffix in [target_new, target_true]
    ],
    padding=True,
    return_tensors="pt",
    ).to("cuda")#如果是llama，开头会是0。但是target部分没有差别

    if use_llama:
        a_tok, b_tok = (tok(f"{n}")["input_ids"][1:] for n in [target_new, target_true])#llama针对target，不要第一个
    else:
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])#不同target的长度
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):#每条prompt--分别拼接了target_new target_true
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]#对应target的第j个token
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0#对应位置的logits的softmax值的ground truth的值
            )[cur_tok].item()
        probs[i] /= cur_len#归一化

        # Compute accuracy on new targets
        #这里是计算预测正确性的。prompt和rephrase都要预测target_new，neighbor预测target_true（每个target token都要预测正确才行）
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(#每个prompt生成长为100的序列，快速生成
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)#所有文本的ngram_entropy的平均--fluency
    #下面才算consistency
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }
    #计算生成essence_texts的困惑度，但好像也没有用到
    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):#算术平均数
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)#n_gram的频率
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A#两个文本向量在tf-idf矩阵中的矩阵表示，再转为2d的稠密表示（基于词0
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()#cos_sim
