import torch
from sympy import print_glsl
from torch.utils.data import Dataset, DataLoader

from project.seq2seq.data_process_func import get_processed_file_data


class PairDataSet(Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.len = len(pairs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 对索引进行边界检查
        idx = min(max(idx, 0), self.len - 1)
        pair = self.pairs[idx]
        x = pair[0]
        y = pair[1]
        # 将文本转为tensor
        x = [self.input_lang.word2index[word] for word in x.split(" ")]
        x.append(self.input_lang.word2index["EOS"])
        y = [self.output_lang.word2index[word] for word in y.split(" ")]
        y.append(self.output_lang.word2index["EOS"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)
        return tensor_x, tensor_y


if __name__ == "__main__":
    input_lang, output_lang, pairs = get_processed_file_data("../../data/english_to_french/data/eng-fra.txt")
    dataset = PairDataSet(input_lang, output_lang, pairs)
    print(dataset[0])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (x, y) in enumerate(data_loader):
        print(x, y)
        if i == 0:
            break
