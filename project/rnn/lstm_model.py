import torch
#导入nn准备构建模型
import torch.nn as nn
class NameClassifyLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers=1):
        super(NameClassifyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden0, cell0):
        input = input.unsqueeze(1) # [sequence_length , 57] -> [sequence_length,batch_size, 57]
        output,(hn,celln) = self.lstm(input, (hidden0, cell0))
        temp = output[-1] #[sequence_length ,batch_size, hidden_size] -> [batch_size, hidden_size]
        return self.softmax(self.linear(temp)) ,hn,celln

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h0,c0