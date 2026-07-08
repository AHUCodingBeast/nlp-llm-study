import torch
import torch.nn as nn

y_true=torch.tensor([1,2],dtype=torch.int64)
y_pred=torch.tensor([[12,20,3],[9,2,14]],dtype=torch.float32)

# CrossEntropyLoss 会同时执行softmax和损失计算
loss = nn.CrossEntropyLoss()
l1=loss(y_pred,y_true).numpy()
print("CrossEntropyLoss",l1)

logsoftmax = nn.LogSoftmax(dim=-1)
print("logsoftmax",logsoftmax(y_pred))
loss = nn.NLLLoss()
# y_true 是 2 的  logsoftmax(y_pred) 是 2*3
# NLLLoss 不需要形状一致 他是拿着 y_true[i] 作为  logsoftmax(y_pred) 对应行的列索引  做交叉熵计算
l2=loss(logsoftmax(y_pred),y_true).numpy()
print("NLLLoss",l2)
