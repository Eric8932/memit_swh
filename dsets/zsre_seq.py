import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"

def process_prompt(ori_str):
    #vicua+3.带着Answer these questions问
    ori_str = ori_str.replace("nq question: ","")
    #Answer these questions:\nQuesion: "+ori_str+"\nAnswer:
    return "Quesion: "+ori_str+"\nAnswer:"

#zsre的prompt rephrase都只有一个，对应同一个target。 neighborhood/loc也只有一个，但是对应一个新的target
#src rephrsae有问号。但是loc无问号 不过loc的开头是"nq question: "
class MENDQADataset_Seq:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """
    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, llama=False,new_prompt=True,*args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_swh_eval.json"
       
        self.llama=llama
        self.new_prompt =new_prompt
        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        if new_prompt:#prompt rephrase loc都用新的prompt。
            for i, record in enumerate(raw):
                #ans都要空一格
                if llama:
                    ans_toks = tok(record["loc_ans"])["input_ids"][1:]
                else:
                    ans_toks = tok(" " + record["loc_ans"])["input_ids"]#loc_ans的id序列
                if self.llama:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": process_prompt(record["loc"]) + " "[:i]+ tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                else:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": process_prompt(record["loc"]) + tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                data.append(
                    {
                        "case_id": i,
                        "requested_rewrite": {
                            "prompt": process_prompt(record["src"].replace(record["subject"], "{}")),#替换掉输入中的subject
                            "subject": record["subject"],
                            "target_new": {"str": record["answers"][0]},#第一个回答
                            "target_true": {"str": "<|endoftext|>"},
                        },
                        #下面这个只在算指标时才会用到
                        "paraphrase_prompts": process_prompt(record["rephrase"]),
                        "neighborhood_prompts": np,
                        "loc_ans":record["loc_ans"],
                        "loc_prompt":[process_prompt(record["loc"])],
                        "attribute_prompts": [],
                        "generation_prompts": [],
                    }
                )
        else:#这是原来的版本
            for i, record in enumerate(raw):
                #ans都要空一格
                if llama:
                    ans_toks = tok(record["loc_ans"])["input_ids"][1:]
                else:
                    ans_toks = tok(" " + record["loc_ans"])["input_ids"]#loc_ans的id序列
                if self.llama:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": record["loc"]+ " "[:i]+ tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                else:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": record["loc"] + tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                data.append(
                    {
                        "case_id": i,
                        "requested_rewrite": {
                            "prompt": record["src"].replace(record["subject"], "{}"),#替换掉输入中的subject
                            "subject": record["subject"],
                            "target_new": {"str": record["answers"][0]},#第一个回答
                            "target_true": {"str": "<|endoftext|>"},
                        },
                        #下面这个只在算指标时才会用到
                        "paraphrase_prompts": record["rephrase"],
                        "neighborhood_prompts": np,
                        "loc_ans":record["loc_ans"],
                        "loc_prompt":[record["loc"]],
                        "attribute_prompts": [],
                        "generation_prompts": [],
                    }
                )
        self._data = data
        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
        

class MENDQADataset_Loc:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """
    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, llama=False,new_prompt=False,*args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_swh_eval.json"
       
        self.llama=llama
        self.new_prompt =new_prompt
        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        if new_prompt:#prompt rephrase loc都用新的prompt。
            for i, record in enumerate(raw):
                #ans都要空一格
                if llama:
                    ans_toks = tok(record["loc_ans"])["input_ids"][1:]
                else:
                    ans_toks = tok(" " + record["loc_ans"])["input_ids"]#loc_ans的id序列
                if self.llama:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": process_prompt(record["loc"]) + " "[:i]+ tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                else:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": process_prompt(record["loc"]) + tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                data.append(
                    {
                        "case_id": i,
                        "neighborhood_prompts": np,
                        "loc_ans":record["loc_ans"],
                        "loc_prompt":[process_prompt(record["loc"])],
                        "attribute_prompts": [],
                        "generation_prompts": [],
                    }
                )
        else:#这是原来的版本
            for i, record in enumerate(raw):
                #ans都要空一格
                if llama:
                    ans_toks = tok(record["loc_ans"])["input_ids"][1:]
                else:
                    ans_toks = tok(" " + record["loc_ans"])["input_ids"]#loc_ans的id序列
                if self.llama:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": record["loc"]+ " "[:i]+ tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                else:
                    np = [#针对ans_toks的每一个，构造一个序列
                            {
                                "prompt": record["loc"] + tok.decode(ans_toks[:i]),
                                "target": tok.decode(ans_toks[i]),
                            }
                            for i in range(len(ans_toks))
                        ]
                data.append(
                    {
                        "case_id": i,
                        "neighborhood_prompts": np,
                        "loc_ans":record["loc_ans"],
                        "loc_prompt":[record["loc"]],
                        "attribute_prompts": [],
                        "generation_prompts": [],
                    }
                )
        self._data = data
        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
