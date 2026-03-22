import torch
from torch import nn
from torch.xpu import device

from project.seq2seq.dataset import get_data_loader


class EncoderRNN(nn.Module):
    """
    input_size: 单词总个数
    hidden_size: 单词的词向量的维度
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # input_size=hidden_size  这里把GRU的隐藏层和hidden_size 设一样
        # 设置batch_first 之后 要求输入为 [batch_size, seq_len, hidden_size]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):

        # (1,6) 假定hidden_size 是256则embedding之后 变为 (1,6,256)
        # 如果我们在定义gru的时候没加batch_first=True 则需要使用embedded.transpose(0, 1) 变为 (6,1,256)
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)


if __name__ == "__main__":
    path  = "../../data/english_to_french/data/eng-fra.txt"
    data_loader = get_data_loader(path)
    input_size = data_loader.dataset.input_lang.n_words
    hidden_size = 256
    encoder = EncoderRNN(input_size, hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到GPU处理，没有GPU可以省略
    encoder.to(device)

    for i, (x, y) in enumerate(data_loader):
        # 进行 前向传播
        output,hn = encoder(x, encoder.init_hidden())
        print(output.shape)
        print(output)
        print(hn.shape)
        print(hn)
        if i==0:
            break
