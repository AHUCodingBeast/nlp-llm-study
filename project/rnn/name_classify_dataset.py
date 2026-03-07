import string

import torch
from torch.utils.data import Dataset


class NameClassDataSet(Dataset):
    # 类变量 letters 存储所有字母 总共 57个
    letters = string.ascii_letters + " .,;'"
    category = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
                'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']

    def __init__(self):
        name_list, country_list = read_data_file()
        self.name_list = name_list
        self.country_list = country_list
        self.len = len(name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 对索引进行边界检查
        idx = min(max(idx, 0), self.len - 1)
        x = self.name_list[idx]
        y = self.country_list[idx]
        # 对于每个名字采用 oneHot 编码标识，一个名字有多个字母，每个字母采用 onehot 编码
        # 矩阵形状是 人名长度 * 字母个数 的矩阵  每一行只有一个数值为 1
        tensor_x = torch.zeros(len(x), len(NameClassDataSet.letters))
        for i, letter in enumerate(x):
            # 找到 x 也就是人名里面每个字母的索引 标记为 1 实现 onehot 编码
            tensor_x[i][NameClassDataSet.letters.find(letter)] = 1
        tensor_y = torch.tensor(self.category.index(y), dtype=torch.long)
        return tensor_x, tensor_y



def read_data_file():
    name_list = []
    country_list = []

    with open("../../data/names.txt", 'r', encoding='utf-8') as f:
        for line in f:

            # split() 会自动处理多个连续空格
            parts = line.strip().split()

            # 确保每行至少有两个部分（姓名和国家）
            if len(parts) >= 2:
                name = parts[0]
                country = parts[1]
                name_list.append(name)
                country_list.append(country)

    return name_list, country_list