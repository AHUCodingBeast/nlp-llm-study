import torch
import numpy as np
import torch
import torch.nn as nn
# import torch.opt im as optim
from torch.utils.data import Dataset, DataLoader, random_split
import re  # 正则表达式处理文本
import matplotlib.pyplot as plt
import seaborn as sns  # 更美观的可视化
from tqdm import tqdm  # 训练进度条
import unicodedata  # 处理特殊字符
import random  # 用于随机生成数据
import torch.nn.functional as F
import time

# 参考链接：
# https://www.cnblogs.com/luzhanshi/articles/18987873

torch.manual_seed(1)
np.random.seed(1)


class Lang:
    # 句子开始标志
    SOS_token = 0
    # 句子结束标志
    EOS_token = 1

    def __init__(self, name):
        """初始化语言对象
        参数:
            name: 语言名称（如'eng'表示英语，'fra'表示法语）
        """
        self.name = name
        self.word2index = {"SOS": Lang.SOS_token, "EOS": Lang.EOS_token}
        self.index2word = {Lang.SOS_token: "SOS", Lang.EOS_token: "EOS"}  # 索引到词汇的映射字典
        self.n_words = 2  # 词汇总数，初始为2（SOS和EOS已占用）

    def addWord(self, word):
        """添加词汇到映射表
        参数:
            word: 要添加的词汇
        """
        # 如果词汇不在映射表中，则添加它
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1  # 词汇计数加1

    def addSentence(self, sentence):
        """将句子拆分为词汇并添加到映射表
        参数:
            sentence: 要处理的句子
        """
        # 按空格分割句子为词汇列表
        for word in sentence.split(' '):
            self.addWord(word)


# 句子最大长度限制
MAX_LENGTH = 10
eng_prefixes = (
    "we are", "we re ",
    "they are", "they re ",
    "i am", "i m ",
    "you are", "you re ",
    "he is", "she is",
    "it is", "we will",
    "they will", "i will"
)





def normalize_string(s):
    """字符串规范化处理
    参数:
        s: 待规范化的字符串
    返回:
        规范化后的字符串
    """
    # 转为小写并去除首尾空格，再去除重音标记
    s = unicode_to_ascii(s.lower().strip())
    # 在标点符号前添加空格，便于后续分割
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中 不是大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def unicode_to_ascii(s):
    """将Unicode字符串转换为ASCII字符
    主要用于去除重音标记等特殊符号
    参数:
        s: 待转换的Unicode字符串
    返回:
        转换后的ASCII字符串
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def filter_pair(p):
    """过滤单个语言对
    参数:
        p: 语言对，格式为[源语言句子, 目标语言句子]
    返回:
        布尔值，True表示符合要求，False表示不符合
    """
    # 检查句子长度是否在限制范围内，且源语言句子是否以指定前缀开头
    return len(p[0].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes) and \
        len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """过滤语言对列表
    参数:
        pairs: 待过滤的语言对列表
    返回:
        过滤后的语言对列表
    """
    return [pair for pair in pairs if filter_pair(pair)]


def get_pairs_from_file(data_path, input_lang, output_lang):
    # data_path = "../../data/english_to_french/data/eng-fra.txt"
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
        # [[英文,法文]]
        pairs = [[normalize_string(s) for s in l.split("\t")] for l in lines]
        # print(pairs[:5])
        input_lang = Lang(input_lang)
        output_lang = Lang(output_lang)
        return input_lang, output_lang, pairs


def get_processed_file_data(path):
    ##path = "../../data/english_to_french/data/eng-fra.txt"
    input_lang, output_lang, pairs = get_pairs_from_file(path, "eng", "fra")
    print(f"原始语对的数量{len(pairs)}")

    # [[英文,法文]]
    pairs = filter_pairs(pairs)
    print(f"过滤之后原始语对的数量{len(pairs)}")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print(f"输入语言的词数{input_lang.n_words}")
    print(f"输出语言的词数{output_lang.n_words}")
    return input_lang, output_lang, pairs
