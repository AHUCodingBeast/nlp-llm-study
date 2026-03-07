import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from itertools import chain

# macOS 使用系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('../data/comment_train.tsv', sep='\t')
# 这里返回的是一个迭代器对象 <class 'map'>
t1 = map(lambda x:list(jieba.cut(x)), train_data['content'])
# * 的作用：将可迭代对象拆解成独立的参数
# *t1 将迭代器拆解成：chain(['房', '间'], ['服', '务'], ['非', '常'])
# chain 将这些列表连接成一个迭代器：['房', '间', '服', '务', '非', '常']
t2 = chain(*t1)
t3 = set(t2)

print(f'训练集的词汇总数为 {len(t3)}')
