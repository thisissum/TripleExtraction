import json
import random
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class CasRelTokenizer(BertTokenizer):
    def _tokenize(self, text):
        return list(text)

class CustomDataset(Dataset):
    def __init__(self, data_path, bert_dir, rela2id, seq_len=200):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.bert_dir = bert_dir
        self.tokenizer = CasRelTokenizer.from_pretrained(bert_dir)
        self.rela2id = rela2id
        self.seq_len = seq_len
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.loads(f.readline())
        self.data = [line for line in self.data if len(line['text']) < seq_len-2 and len(line['spo_list']) > 0]

    def __len__(self):
        return len(data)

    def __getitem__(self, item):
        line = self.data[item]
        text, spo_list = line['text'], line['spo_list']

        s2po = defaultdict(lambda: [])
        for spo in spo_list:
            s2po[spo['subject']].append({'predicate': spo['predicate'], 'object': spo['object']})

        tk_output = self.tokenizer.encode_plus(
            text=text,
            max_length=self.seq_len,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        token_ids, mask, segment_ids = map(
            lambda x: x.squeeze(),
            (tk_output['input_ids'], tk_output['attention_mask'], tk_output['token_type_ids'])
        )

        golden_sub = torch.FloatTensor(np.zeros((self.seq_len, self.seq_len)))
        for spo in spo_list:
            sub = spo['subject']
            sub_start_idx = text.find(sub) + 1
            sub_end_idx = sub_start_idx + len(sub) - 1
            golden_sub[sub_start_idx, sub_end_idx] = 1

        random_sub = random.choice(list(s2po.keys()))
        sub_start = torch.LongTensor([text.find(random_sub) + 1])
        sub_end = torch.LongTensor([sub_start + len(random_sub) - 1])
        golden_obj = torch.FloatTensor(np.zeros((self.seq_len, self.seq_len, len(self.rela2id))))
        for po in s2po[random_sub]: # given one subject, predict all objects and relas
            rela, obj = po['predicate'], po['object']
            obj_start = text.find(obj) + 1  # + 1 for cls
            obj_end = obj_start + len(obj) - 1
            rela_id = self.rela2id[rela]
            golden_obj[obj_start, obj_end, rela_id] = 1

        return {
            "token_ids": token_ids,
            "mask": mask,
            "segment_ids": segment_ids,
            "golden_sub": golden_sub,
            "sub_start": sub_start,
            "sub_end": sub_end,
            "golden_obj": golden_obj
        }
