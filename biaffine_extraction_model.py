import torch
from torch import nn
from transformers import BertModel
from .module.layer import BiaffineTagger

class BiaffineExtractor(nn.Module):
    def __init__(self, rela_num, hidden_size=768, bert_dir='./bert'):
        super(BiaffineExtractor, self).__init__()
        self.output_size = output_size
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
        self.object_tagger = BiaffineTagger(hidden_size=hidden_size, output_size=rela_num)


    def forward(self, inputs):
        sub_out, hidden = self.forward_sub(
            token_ids=inputs["token_ids"],
            mask=inputs["mask"],
            segment_ids=inputs["segment_ids"]
        )
        obj_out = self.forward_obj(
            hidden=hidden,
            batch_start_idx=inputs["batch_start_idx"].squeeze(),
            batch_end_idx=inputs["batch_end_idx"].squeeze()
        )
        return sub_out, obj_out


    def forward_sub(self, token_ids, mask=None, segment_ids=None):
        if self.bert_dir is not None:
            hidden = self.embedding_layer(token_ids, mask, segment_ids)[0]
        else:
            hidden = self.embedding_layer(token_ids)
        sub_out = self.subject_tagger(hidden).squeeze(-1)
        return sub_out, hidden


    def forward_obj(self, hidden, batch_start_idx, batch_end_idx):
        batch_size = hidden.size(0)
        device = hidden.device
        batch_idx = torch.arange(batch_size).to(device)
        # select (0, 5), (1, 3)。。。 where first dim is batch and second dim is time_step at seq_len
        sub_start_hidden = hidden[batch_idx, batch_start_idx]
        sub_end_hidden = hidden[batch_idx, batch_end_idx]
        # shape(v_sub) = batch_size, hidden_size
        v_sub = (sub_start_hidden + sub_end_hidden) / 2 # according to source code from keras version
        hidden = hidden + v_sub.unsqueeze(1)
        obj_out = self.object_tagger(hidden)
        return obj_out