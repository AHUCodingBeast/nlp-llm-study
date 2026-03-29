# teacher_forcing介绍
# 它是一种用于序列生成任务的训练技巧，在seq2seq架构中，根据循环神经网络理论，
# 解码器每次应该使用上一步的结果作为输入的一部分，
# 但是训练过程中，一旦上一步的结果是错误的，就会导致这种错误被累积，无法达到训练效果，因此，我们需要一种机制改变上一步出错的情况，
# 因为训练时我们是已知正确的输出应该是什么，因此可以强制将上一步结果设置成正确的输出这种方式就叫做teacher_forcing.
import random

import torch
from torch import nn

from project.seq2seq.AttentionDecoderRNN import AttentionDecoderRNN
from project.seq2seq.DecoderRNN import DecoderRNN
from project.seq2seq.EncoderRNN import EncoderRNN
from project.seq2seq.data_process_func import Lang
from project.seq2seq.dataset import get_data_loader


# 能够在训练的时候矫正模型的预测，避免在序列成的过程中误差进一步放大.
# teacher_forcing能够极大的加快模型的收敛速度，令模型训练过程更快更平稳.

def iter_training(x, y, encoder: EncoderRNN, decoder: AttentionDecoderRNN, encode_adam, decode_adam, criterion,
                  teach_force_ratio):
    # [1,6],[1,1,256] -> [1,6,256],[1,1,256]
    encoder_output, encoder_hidden = encoder(x, encoder.init_hidden())

    # [MAX_LENGTH,256] 也就是 [10,256]
    v = torch.zeros(AttentionDecoderRNN.MAX_LENGTH, encoder.hidden_size)
    for idx in range(x.shape[1]):
        v[idx] = encoder_output[0, idx]

    decode_hidden = encoder_hidden

    input_y = torch.tensor([[Lang.SOS_token]])

    my_loss = 0.0
    # 法文的句子长度
    y_len = y.shape[1]

    use_teach_force = True if random.random() < teach_force_ratio else False
    if use_teach_force:
        for idx in range(y_len):
            # [1,1],[1,1,256],[10,256] --> [1,4355],[1,1,256],[1,10]
            output_y, decode_hidden, weight = decoder(input_y, decode_hidden, v)
            target_y = y[0][idx].view(1)
            my_loss += criterion(output_y, target_y)
            input_y = y[0][idx].view(1, -1)
        else:
            for idx in range(y_len):
                output_y, decode_hidden, weight = decoder(input_y, decode_hidden, v)
                target_y = y[0][idx].view(1)
                my_loss += criterion(output_y, target_y)
                topv, topi = output_y.topk(1, dim=-1)
                if topi.item() == Lang.EOS_token:
                    break
                input_y = topi.detach()
    decode_adam.zero_grad()
    encode_adam.zero_grad()
    my_loss.backward()
    decode_adam.step()
    encode_adam.step()
    return my_loss.item() / y_len


def train():
    lr = 1e-4
    epoch = 2
    teach_force_ratio = 0.5

    path = "../../data/english_to_french/data/eng-fra.txt"
    data_loader = get_data_loader(path)

    # 实例化编码器和解码器
    hidden_size = 256
    encoder = EncoderRNN(data_loader.dataset.input_lang.n_words, hidden_size)
    decoder = AttentionDecoderRNN(data_loader.dataset.output_lang.n_words, hidden_size)

    # 优化器
    adam_encoder = torch.optim.Adam(encoder.parameters(), lr=lr)
    adam_decoder = torch.optim.Adam(decoder.parameters(), lr=lr)

    # 损失函数
    criterion = nn.NLLLoss()

    plot_loss_list = []

    # 训练
    for epoch_idx in range(1, epoch + 1):
        print_loss_total = 0
        plot_loss_total = 0

        for i, (x, y) in enumerate(data_loader):
            loss = iter_training(x, y, encoder, decoder, adam_encoder, adam_decoder, criterion, teach_force_ratio)
            print_loss_total += loss
            plot_loss_total += loss
