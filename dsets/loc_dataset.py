import os
import jsonlines
import random
import torch
from torch.utils.data import Dataset
import json

def process_prompt(ori_str):
    #vicua+3.带着Answer these questions问
    #Answer these questions:\nQuesion: "+ori_str+"\nAnswer:
    return "Quesion: "+ori_str+"\nAnswer:"

#这里要保证是left_pad
#只用于loc
class ZSRE_Loc(Dataset):
    def __init__(self, tokenizer, data_path, max_length=128, dataset_size=1000):
        """
        :param tokenizer:
        :param data_path:
        :param max_length:
        :param validation:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.size = dataset_size
        self.max_length = max_length

        data_path = os.path.join(data_path, 'zsre_swh_eval.json')
        with open(data_path,'r') as f:
            all_d = json.load(f)
            for case_id,d in enumerate(all_d):
                d1 = {}
                d1['case_id'] = case_id
                d1['input'] = process_prompt(d['loc'])#已经处理过了
                d1['output'] = d['loc_ans']
                d1['rephrases']  = []
                    
                self.data.append(d1)
        
        self.data = self.data[:self.size]
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["output"],
            "rephrases": self.data[item]["rephrases"],
            'case_id':self.data[item]['case_id']
                
        }

    def collate_fn(self, batch):
        batches = {}
        for name in ("src",):#validation-(src), (src,trg)
            tokenizer_input = [b[name] for b in batch]
            tokenizer_output = self.tokenizer(
                tokenizer_input, return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_output.items():#input_ids 和 attention_mask
                batches["{}_{}".format(name, k)] = v
        batches["raw"] = batch
        return batches


class MCF_Loc(Dataset):
    def __init__(self, tokenizer, data_path, max_length=128, dataset_size=1000):
        """
        :param tokenizer:
        :param data_path:
        :param max_length:
        :param validation:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.size = dataset_size
        self.max_length = max_length

        #分为两类，1.edit--src,trg,rephrsae都需要，其中还专门有一个数据集，只要loc和gene（这里没有gene）
        data_path = os.path.join(data_path, 'multi_counterfact.json')
        with open(data_path,'r') as f:
            all_d = json.load(f)
            for i,d in enumerate(all_d):
                d1 = {}
                d1['input'] = d['neighborhood_prompts']#之前这里只有一条，现在是一个列表
                d1['output'] = d['requested_rewrite']['target_true']['str']
                d1['rephrases']  = []
                d1['gene'] = d['generation_prompts']
                d1['relation_id'] = d["requested_rewrite"]["relation_id"]
                d1['target_new_id'] = d['requested_rewrite']['target_new']['id']
                d1['subject'] = d['requested_rewrite']['subject']
                d1['case_id'] = d['case_id']
                    
                self.data.append(d1)
        
        self.data = self.data[:self.size]
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["output"],
            "rephrases": self.data[item]["rephrases"],
            "gene": self.data[item]["gene"],
            "relation_id": self.data[item]["relation_id"],
            "target_new_id": self.data[item]["target_new_id"],
            "subject": self.data[item]["subject"],
            'case_id':self.data[item]['case_id']
        }

    #还是分成edit，memory和loc。edit需要训练，所以
    def collate_fn(self, batch):
        batches = {}
        for name in ("src",):#validation-(src), (src,trg)
            tokenizer_input = [sin_src for b in batch for sin_src in b[name]]
            tokenizer_output = self.tokenizer(
                tokenizer_input, return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_output.items():#input_ids 和 attention_mask
                batches["{}_{}".format(name, k)] = v
        batches["raw"] = batch
        return batches

