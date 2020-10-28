import torch
from torch import nn

class BiaffineTagger(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(BiaffineTagger, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.start_fc = nn.Linear(hidden_size, hidden_size)
        self.end_fc = nn.Linear(hidden_size, hidden_size)
        self.biaffine_weight = nn.Parameter(
            torch.nn.init.orthogonal_(torch.Tensor(hidden_size, output_size, hidden_size))
        )
        self.concat_weight = nn.Linear(hidden_size, output_size)

    def forward(self, hidden):
        """

        :param inputs: torch.FloatTensor, shape(inputs) = (batch_size, seq_len, hidden_size)
        :return: output: torch.FloatTensor, shape(output) = (batch_size, seq_len, 1, seq_len)
        """
        hidden_start = self.start_fc(hidden)
        hidden_end = self.end_fc(hidden)
        biaffine_out = hidden_start.matmul(self.biaffine_weight).matmul(hidden_end.permute(0,2,1))
        # shape(biaffine_out) = batch_size, seq_len, output_size, seq_len
        concat_out = self.concat_weight(torch.cat([hidden_start, hidden_end], dim=-1))
        # shape(concat_out) = batch_size, seq_len, output_size
        output = (biaffine_out + concat_out).permute(0,1,3,2)

        return output


