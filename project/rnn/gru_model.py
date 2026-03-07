import torch
#导入nn准备构建模型
import torch.nn as nn
class NameClassifyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(NameClassifyGRU, self).__init__()
        # 输入层大小 在这里其实就是一个单词的向量维度 这个案例里面就是 57
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 输出层大小 在这里就是对应的一个国家的分类结果 对应的向量形状是 （18，）
        self.output_size = output_size
        self.num_layers = num_layers
        # GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):




        # gru 要求的 input 的 shape = [sequence_length , batch_size, input_size]
        # sequence_length: 输入序列长度,也就是句子的长度
        # batch_size: 批次样本的数量
        # input_size: 输入张量 x 的维度，也就是每个词的维度

        # 原始的 input 形状是 [sequence_length , 57]，需要补齐 batch_size
        # [sequence_length , 57] -> [sequence_length ,1, 57]
        input = input.unsqueeze(1)
        output,hn = self.gru(input, hidden)
        return self.softmax(self.linear(output[-1])) ,hn

    def init_hidden(self):
       # 初始化隐藏层
       # 参数 1：num_layers =RNN 结构体包含的层数
       # 参数 2：batch_size=批次样本的数量
       # 参数 3：hidden_size=隐藏层输出 h 的维度 也就是隐藏层的神经元个数
        return torch.zeros(self.num_layers, 1, self.hidden_size)