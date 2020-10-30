import torch
from torch import nn

class BiaffineTagger(nn.Module):
    def __init__(self, hidden_size):
        super(BiaffineTagger, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.start_fc = nn.Linear(hidden_size, hidden_size)
        self.end_fc = nn.Linear(hidden_size, hidden_size)
        self.biaffine_weight = nn.Parameter(
            torch.nn.init.orthogonal_(torch.Tensor(hidden_size, hidden_size))
        )
        self.concat_weight = nn.Linear(hidden_size*2, 1)

    def forward(self, hidden):
        """

        :param inputs: torch.FloatTensor, shape(inputs) = (batch_size, seq_len, hidden_size)
        :return: output: torch.FloatTensor, shape(output) = (batch_size, seq_len, 1, seq_len)
        """
        hidden_start = self.start_fc(hidden)
        hidden_end = self.end_fc(hidden)
        biaffine_out = hidden_start.matmul(self.biaffine_weight).matmul(hidden_end.permute(0,2,1))
        # shape(biaffine_out) = batch_size, seq_len, seq_len
        concat_out = self.concat_weight(torch.cat([hidden_start, hidden_end], dim=-1))
        # shape(concat_out) = batch_size, seq_len, 1
        output = (biaffine_out + concat_out)
        # shape(output) = batch_size, seq_len, seq_len
        return torch.sigmoid(output)

class ObjectTagger(nn.Module):
    """
    Predict object from a given subject
    """
    def __init__(self, hidden_size, rela_num):
        super(ObjectTagger, self).__init__()
        self.hidden_size = hidden_size
        self.rela_num = rela_num

        self.start_fc = nn.Linear(hidden_size, rela_num)
        self.end_fc = nn.Linear(hidden_size, rela_num)

    def forward(self, hidden, batch_start_idx, batch_end_idx):
        """
        :param hidden: torch.FloatTensor, shape = batch_size, seq_len, hidden_size
        :param batch_start_idx: torch.LongTensor, shape = batch_size
        :param batch_end_idx: torch.LongTensor, shape = batch_size
        :return: (obj_start_pointer, obj_end_pointer), shape = batch_size, seq_len
        """
        batch_size = hidden.size(0)
        device = hidden.device
        batch_idx = torch.arange(batch_size).to(device)
        # select (0, 5), (1, 3)。。。 where first dim is batch and second dim is time_step at seq_len
        sub_start_hidden = hidden[batch_idx, batch_start_idx]
        sub_end_hidden = hidden[batch_idx, batch_end_idx]
        # shape(v_sub) = batch_size, hidden_size
        v_sub = (sub_start_hidden + sub_end_hidden) / 2 # according to source code from keras version
        hidden = hidden + v_sub.unsqueeze(1)
        start_pointer = torch.sigmoid(self.start_fc(hidden))
        end_pointer = torch.sigmoid(self.end_fc(hidden))
        return start_pointer, end_pointer
