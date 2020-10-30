import torch
from torch import nn
from transformers import BertModel
from module.layer import BiaffineTagger, ObjectTagger

class BiaffineExtractor(nn.Module):
    def __init__(self, rela_num, hidden_size=768, bert_dir='./bert'):
        super(BiaffineExtractor, self).__init__()
        self.rela_num = rela_num
        self.hidden_size = hidden_size
        self.bert_dir = bert_dir

        if bert_dir is None:
            self.embedding_layer = nn.Sequential(
                nn.Embedding(24000, hidden_size, padding_idx=0),
                nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2, bidirectional=True, batch_first=True)
            )
        else:
            self.embedding_layer = BertModel.from_pretrained(bert_dir)

        self.subject_tagger = BiaffineTagger(hidden_size=hidden_size, output_size=1)
        self.object_tagger = ObjectTagger(hidden_size=hidden_size, rela_num=rela_num)


    def forward(self, inputs):
        sub_out, hidden = self.forward_sub(
            token_ids=inputs["token_ids"],
            mask=inputs["mask"],
            segment_ids=inputs["segment_ids"]
        )
        obj_head_pred, obj_tail_pred = self.forward_obj(
            hidden=hidden,
            batch_start_idx=inputs["sub_start"].squeeze(),
            batch_end_idx=inputs["sub_end"].squeeze()
        )
        return sub_out, obj_head_pred, obj_tail_pred


    def forward_sub(self, token_ids, mask=None, segment_ids=None):
        if self.bert_dir is not None:
            hidden = self.embedding_layer(token_ids, mask, segment_ids)[0]
        else:
            hidden = self.embedding_layer(token_ids)[0]
        sub_out = self.subject_tagger(hidden).squeeze(-1)
        return sub_out, hidden


    def forward_obj(self, hidden, batch_start_idx, batch_end_idx):
        obj_head_pred, obj_tail_pred = self.object_tagger(hidden, batch_start_idx, batch_end_idx)
        return obj_head_pred, obj_tail_pred