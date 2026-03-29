import torch.nn.functional as F
from torch import nn

from project.seq2seq.EncoderRNN import EncoderRNN
from project.seq2seq.dataset import get_data_loader


# 解码器
# 解码器的输入就是法文 【本实例演示的是没有注意力机制的Decoder模型】

class DecoderRNN(nn.Module):
    """
    input_size: 法文单词总个数
    hidden_size: 法文单词的词向量的维度
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 为什么这里是inputsize？ 首先这里的inputSize代表的是法文单词的总数
        # 我们经过线性层处理之后实际上最终我们想要的就是每个单词的概率 所以自然而然的就要输出法文单词的个数个维度
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input 的形状是(1,1) -> (1,1,256), 每次送进去一个法文单词
        output = self.embedding(input)
        # 添加relu层使得embedding矩阵更加稀疏 防止过拟合
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # (1,1,256) -> (1,256) -> (1,4345(output_size))
        output = self.softmax(self.out(output[0]))
        return output, hidden


if __name__ == "__main__":
    path = "../../data/english_to_french/data/eng-fra.txt"
    data_loader = get_data_loader(path)

    english_word_size = data_loader.dataset.input_lang.n_words
    hidden_size = 256
    encoder = EncoderRNN(english_word_size, hidden_size)
    print(f"编码器架构={encoder}")
    # encoder.to( device)

    french_word_size = data_loader.dataset.output_lang.n_words
    decoder = DecoderRNN(french_word_size, hidden_size)
    print(f"解码器架构={decoder}")

    for i, (x, y) in enumerate(data_loader):
        # 进行 前向传播

        encode_output, hn = encoder(x, encoder.init_hidden())
        print('编码器输出 >> ', encode_output.shape, hn.shape)

        # tensor([[  6,  11,  65, 870, 299,   5,   1]]) torch.Size([1, 7])
        print(y, y.shape)

        # 解码的时候是一个字符一个字符的送进去的
        for k in range(y.shape[1]):
            # print(y[0][k].shape, y[0][k].view(1, -1).shape)
            temp = y[0][k].view(1, -1)
            # hn 直接用编码器的隐藏层输出
            output, hn = decoder(temp, hn)
            print('解码器输出 >> ', output.shape, hn.shape)
        if i == 0:
            break
