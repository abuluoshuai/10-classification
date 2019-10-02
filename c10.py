import os
import torch
import random
import pickle
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
import matplotlib.pyplot as plt

EPOCH = 10
BATCH_SIZE = 4000

train_data_list = []
train_labels_list = []

data_filenames = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
for filename in data_filenames:
    with open(os.path.join("data", filename),"rb") as f:
        data1 = pickle.load(f, encoding='bytes')
    train_data_list.append(torch.tensor(data1[b"data"].reshape((-1,3,32,32))).type(torch.FloatTensor)/255.)
    train_labels_list.append(torch.tensor(data1[b"labels"]).type(torch.LongTensor))

train_data = torch.cat(train_data_list)
train_labels = torch.cat(train_labels_list)

print(max(train_labels),min(train_labels))

with open(os.path.join("data/test_batch"),"rb") as f:
    test = pickle.load(f, encoding='bytes')

test_data = torch.tensor(test[b"data"].reshape((-1,3,32,32))).type(torch.FloatTensor).cuda()/255.
test_labels = torch.tensor(test[b"labels"]).type(torch.LongTensor).cuda()

train_set = Data.TensorDataset(train_data,train_labels)
train_loader = Data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# plt.imshow(data1[b'data'].reshape((-1,32,32,3),order = "F")[1000])
# plt.title(data1[b"labels"][1000])
# plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,15,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(15,30,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(30*8*8,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        r = self.out(x)
        return r

cnn = CNN()
cnn.cuda()
opt = torch.optim.Adam(cnn.parameters(),lr = 0.000001)
loss_f = nn.CrossEntropyLoss()
es = []
loss_list = []
acc_list = []

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        # print(x.shape,y.shape,y)
        b_x = x.cuda()
        b_y = y.cuda()
        pre = cnn(b_x)
        loss = loss_f(pre,b_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 1 == 0:
        # print("Loss:%.4f"%loss.cpu().data.numpy())
        test_output = cnn(test_data)
        pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

        accuracy = torch.sum(pred_y == test_labels).type(torch.FloatTensor) / test_labels.size(0)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy.numpy())
        es.append(epoch/100)
        loss_list.append(float(loss.data.cpu().numpy()))
        acc_list.append(float(accuracy.numpy()))

# print(es,acc_list,loss_list)
torch.save(cnn, "cnn-10.pkl")
plt.figure(figsize = (12,14))
plt.subplot(2,1,1)
plt.title("Loss")
plt.plot(es, loss_list, "r-.", label = "loss")
plt.legend()
plt.subplot(2,1,2)
plt.title("Accuracy")
plt.plot(es, acc_list, "r-.", label = "accuracy")
plt.legend()
plt.show()