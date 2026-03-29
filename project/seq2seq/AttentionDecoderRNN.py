import torch
import torch.nn.functional as F
from torch import nn

from project.seq2seq.EncoderRNN import EncoderRNN
from project.seq2seq.dataset import get_data_loader


class AttentionDecoderRNN(nn.Module):
    # 英文单词的最大长度 ，这里如果不满足这个长度需要做填充，否则做矩阵乘法bmm会有问题
    MAX_LENGTH = 10

    def __init__(self, vocab_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH, device=None):
        super(AttentionDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 输出 1,1,256
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # Q和K进行拼接
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, q, k, v):
        # 输入的形状是(1,1) -> (1,1,256), 每次送进去一个法文单词
        q = self.embedding(q)
        q = self.dropout(q)

        # 拼接Q和K拼接计算相似度 1,256 cat 1,256 -> 1,512 经过attn -> 1,10
        catted = torch.cat((q[0], k[0]), dim=-1)
        weights = F.softmax(self.attn(catted), 1)
        # 使用升维unsqueeze(0) -> 1,1,10  *  1,10,256  = 1,1,256
        # 权重和V进行矩阵乘法得到上下文向量
        c = torch.bmm(weights.unsqueeze(0), v.unsqueeze(0))
        # q 和 c进行拼接 1,256 cat 1,256 -> 1,512
        o = torch.cat((q[0], c[0]), 1)
        # 拼接之后的结果送入线性层
        o = self.attn_combine(o).unsqueeze(0)
        o = F.relu(o)
        output, k = self.gru(o, k)
        output = self.softmax(self.out(output[0]))
        return output, k, weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


if __name__ == "__main__":
    path = "../../data/english_to_french/data/eng-fra.txt"
    data_loader = get_data_loader(path)

    english_word_size = data_loader.dataset.input_lang.n_words
    hidden_size = 256
    encoder = EncoderRNN(english_word_size, hidden_size)
    print(f"编码器架构={encoder}")

    french_word_size = data_loader.dataset.output_lang.n_words
    decoder = AttentionDecoderRNN(french_word_size, hidden_size)
    print(f"解码器架构={decoder}")

    for i, (x, y) in enumerate(data_loader):
        encode_output, hn = encoder(x, encoder.init_hidden())
        # 1,6,256 1,1,256
        print('编码器输出 >> ', encode_output.shape, hn.shape)

        #  v shape=(10, 256)
        v = torch.zeros(AttentionDecoderRNN.MAX_LENGTH, encoder.hidden_size)
        # 将v的值设置为编码器的输出 不足长度的以0填充
        for idx in range(y.shape[1]):
            v[idx] = encode_output[0, idx]

        # v= v.unsqueeze(0)

        for idx in range(y.shape[1]):
            # 解码器输入
            q = y[0][idx].view(1, -1)
            # 编码器的隐藏层输出
            k = hn
            output, hn, weights = decoder(q, k, v)
            print('解码器输出 >> ', output.shape, hn.shape, weights.shape)

        break
