import torch
# import numpy as np
import pandas as pd
from torch import nn
import torchvision
from torchvision import transforms
# from matplotlib import pyplot as plt
from torchsummary import summary
import datetime
from sklearn.metrics import accuracy_score

'''定义一个打印时间的函数'''


def printbar():
    nowtime = datetime.datetime.now().strftime('%H:%M:%S')
    print('========' * 8 + '%s' % nowtime)


'''
读取数据集
'''
transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root=r'dataset', train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=10, shuffle=True)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=10, shuffle=True)
'''
可视化数据
'''

'''脚本风格训练'''

net = nn.Sequential()
net.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3))
net.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
net.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5))
net.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
net.add_module('dropout', nn.Dropout2d(p=0.1))
net.add_module('adaptive_pool', nn.AdaptiveMaxPool2d((1, 1)))
net.add_module('flatten', nn.Flatten())
net.add_module('linear1', nn.Linear(64, 32))  # 需要进行计算
net.add_module('Relu', nn.ReLU())
net.add_module('linear2', nn.Linear(32, 10))

summary(net.to('cuda:0'), input_size=(1, 32, 32))


def accuracy(y_pred, y_label):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_label, y_pred_cls)


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
metric_fun = accuracy
metric_name = 'accuracy'

print('training start....')
epochs = 2
log_step_freq = 100
df = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val' + metric_name])

for epoch in range(1, epochs + 1):
    net.train()
    loss_sum = 0
    metric_sum = 0
    step = 1
    for step, (features, labels) in enumerate(dl_train, 1):
        features, labels = features.cuda(), labels.cuda()
        optimizer.zero_grad()
        predicts = net(features)
        loss = loss_func(predicts, labels)
        metric = metric_fun(predicts.cpu(), labels.cpu())

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step_freq == 0:
            print(('[step=%d] loss:%.3f,' + metric_name + ":%0.3f") % (step, loss_sum / step, metric_sum / step))

    net.eval()
    val_loss_sum = 0
    val_metric_sum = 0
    val_step = 1
    for val_step, (features, labels) in enumerate(dl_train):
        features, labels = features.cuda(), labels.cuda()
        with torch.no_grad():
            val_predicts = net(features)
            val_loss = loss_func(val_predicts, labels)
            val_metric = metric_fun(val_predicts.cpu(), labels.cpu())

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / step, val_metric_sum / step)
    df.loc[epoch - 1] = info
    print(('EPOCH:%d loss:%.3f ' + metric_name + ':%.3f ,val_loss=%.3f' + 'val' + metric_name + ':%.3f') % info)
print('train end.')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# net=Net()
# summary(net.cuda(),input_size=(1,32,32))
