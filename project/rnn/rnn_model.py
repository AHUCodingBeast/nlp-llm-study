import torch
#导入nn准备构建模型
import torch.nn as nn
class NameClassifyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(NameClassifyRNN, self).__init__()

        # 在当前的案例场景下就是一个字母的输入
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.run = nn.RNN(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # sequence_length: 输入序列长度,也就是句子的长度
        # batch_size: 批次样本的数量
        # input_size: 输入张量 x 的维度，也就是每个词的维度
        # 原始的 input 形状是 [sequence_length , 57]，需要补齐 batch_size 这里的 batch_size 对于我们这个人名分类的场景固定为 1
        # [sequence_length , 57] -> [sequence_length ,1, 57]
        input = input.unsqueeze(1)

        # output: [sequence_length , 1, hidden_size]
        # hn: [num_layers, 1, hidden_size]
        output,hn = self.run(input, hidden)

        # 获取RNN最后一个时间步的输出   [1, hidden_size]
        temp = output[-1]

        return self.softmax(self.linear(temp)), hn

    def init_hidden(self):
       # 初始化隐藏层
       # 参数 1：num_layers =RNN 结构体包含的层数
       # 参数 2：batch_size =批次样本的数量
       # 参数 3：hidden_size=隐藏层输出 h 的维度 也就是隐藏层的神经元个数
        return torch.zeros(self.num_layers, 1, self.hidden_size)