from torch import nn, optim
from torch.utils.data import DataLoader
import torch

from project.rnn.gru_model import NameClassifyGRU
from project.rnn.lstm_model import NameClassifyLSTM
from project.rnn.name_classify_dataset import NameClassDataSet
from project.rnn.rnn_model import NameClassifyRNN


def train_check():
    data_loader = DataLoader(NameClassDataSet(), batch_size=1, shuffle=True)
    input_size = len(NameClassDataSet.letters) # 57
    n_hidden = 128
    output_size = len(NameClassDataSet.category) # 多少个国家 目前数据集中 18个国家 对应的分类结果有 18种

    gru = NameClassifyGRU(input_size, n_hidden, output_size)
    rnn = NameClassifyRNN(input_size, n_hidden, output_size)
    lstm = NameClassifyLSTM(input_size, n_hidden, output_size)

    for i, (x, y) in enumerate(data_loader):
        rnn_out, rnn_hn = rnn(x[0], rnn.init_hidden())
        print('RNN OutPut >> ',rnn_out.shape, rnn_hn.shape, rnn_out)
        if i==0:
            break

    for i, (x, y) in enumerate(data_loader):
        lstm_out, lstm_hn, lstm_cell = lstm(x[0], *lstm.init_hidden())
        print('LSTM OutPut >> ',lstm_out.shape, lstm_hn.shape, lstm_cell.shape, lstm_out)
        if i==0:
            break

    for i, (x, y) in enumerate(data_loader):
        gru_out, gru_hn = gru(x[0], gru.init_hidden())
        print('GRU OutPut >> ',gru_out.shape, gru_hn.shape, gru_out)
        if i==0:
            break



def train_gru():

    data_loader = DataLoader(NameClassDataSet(), batch_size=1, shuffle=True)

    input_size = len(NameClassDataSet.letters) # 57
    n_hidden = 128
    output_size = len(NameClassDataSet.category) # 多少个国家 目前数据集中 18个国家 对应的分类结果有 18种

    model = NameClassifyGRU(input_size, n_hidden, output_size)
    # 损失函数
    loss = nn.NLLLoss()
    # 优化器 adam 梯度下降
    adam = optim.Adam(model.parameters(), lr=0.001)
    epochs=1

    for epoch in range(epochs):
        total_loss = 0
        predict_right_count = 0
        for i,(x,y) in enumerate(data_loader):
            out,hn = model(x[0], model.init_hidden())
            cur_loss = loss(out,y)

            adam.zero_grad()
            cur_loss.backward()
            adam.step()

            total_loss += cur_loss.item()
            if out.argmax(1).item() == y:
                predict_right_count += 1

            if i % 5 == 0 and i != 0:
                avg_loss = total_loss / i
                avg_accuracy = predict_right_count / i
                print("epoch:%d, avg_loss:%.3f, avg_accuracy:%.3f" % (epoch, avg_loss, avg_accuracy))


    torch.save(model.state_dict(), "name_classify_gru.pth") # 保存模型




def predict_by_gru(x):
    input_size = len(NameClassDataSet.letters) # 57
    n_hidden = 128
    output_size = len(NameClassDataSet.category) # 多少个国家 目前数据集中 18个国家 对应的分类结果有 18种

    # 声明模型
    gru = NameClassifyGRU(input_size, n_hidden, output_size)
    # 加载模型参数
    gru.load_state_dict(torch.load("name_classify_gru.pth"))

    # x 是一段文本需要改为张量 例如 deng 应该转为 4*57 的张量形式
    tensor_x = torch.zeros(len(x),len(NameClassDataSet.letters))
    for i,c in enumerate(x):
        tensor_x[i][NameClassDataSet.letters.find(c)] = 1

    # 预测的时候不需要做反向传播 和梯度更新
    with torch.no_grad():
        output,hn=gru(tensor_x, gru.init_hidden())
        # print(output.shape) # torch.Size([1, 18])
        # 从预测结果里面获取前 3 个概率最大的 排序依据是第 1 维， True 表示取最大值
        topn = 3
        topv,topi=output.topk(topn,1,True)
        print("x=", x)
        for i in range(topn):

            print(NameClassDataSet.category[topi[0][i]],topv[0][i])



if __name__ == '__main__':
    # train_gru()
    predict_by_gru("deng")