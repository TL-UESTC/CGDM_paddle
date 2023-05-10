from data_loader.folder import ImageFolder_ind
import paddle
import paddle.nn.functional as F
import paddle as T
import paddle.nn as nn
from paddle.autograd import PyLayer
from x2paddle.project_convertor.pytorch.models.resnet import Bottleneck
"""nll_loss = paddle.nn.loss.NLLLoss()
log_softmax = paddle.nn.LogSoftmax(axis=1)

input = paddle.to_tensor([[0.88103855, 0.9908683, 0.6226845],
                          [0.53331435, 0.07999352, 0.8549948],
                          [0.25879037, 0.39530203, 0.698465],
                          [0.73427284, 0.63575995, 0.18827209],
                          [0.05689114, 0.0862954, 0.6325046]], "float32")
log_out = log_softmax(input)
label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
result = nll_loss(log_out, label)
print(result)

pinput = T.tensor([[0.88103855, 0.9908683, 0.6226845],
                   [0.53331435, 0.07999352, 0.8549948],
                   [0.25879037, 0.39530203, 0.698465],
                   [0.73427284, 0.63575995, 0.18827209],
                   [0.05689114, 0.0862954, 0.6325046]])
plable = T.tensor([0, 2, 1, 1, 0])
softmax = nn.LogSoftmax()
plogout = softmax(pinput)
presult = F.nll_loss(plogout, plable)
print(presult)"""
"""class cus_tanh(PyLayer):
    @staticmethod
    def forward(self, x, func1, func2=paddle.square):
        # self is a context object that store some objects for backward.
        self.func = func2
        y = func1(x)
        # Pass tensors to backward.
        self.save_for_backward(y)
        return y

    @staticmethod
    # forward has only one output, so there is only one gradient in the input of backward.
    def backward(self, dy):
        # Get the tensors passed by forward.
        y, = self.saved_tensor()
        grad = dy * (1 - self.func(y))
        # forward has only one input, so only one gradient tensor is returned.
        return grad


data = paddle.to_tensor([[1,2,3],[4,5,6]], dtype="float64")
data.stop_gradient = False
z = cus_tanh.apply(data, func1=paddle.tanh)
z.mean().backward()

print(data.grad)"""
"""data = paddle.ones(shape=[3, 1, 2], dtype='float32')
x = paddle.nn.initializer.XavierNormal()
x(data)
data = T.ones([3,1,2])
nn.init.xavier_normal(data)
# inear.weight:  [[ 0.06910077 -0.18103665]
#                 [-0.02546741 -1.0402188 ]]
# linear.bias:  [-0.5012929   0.12418364]


print(data)
# res:  [[[-0.4576595 -1.0970719]]
#        [[-0.4576595 -1.0970719]]
#        [[-0.4576595 -1.0970719]]]"""
"""a=T.tensor([])
print(a.shape==T.Size([0]))
data = paddle.to_tensor([])
print(data.shape)"""

'''from x2paddle import models
# 构造权重随机初始化的模型：
resnet18 = models.resnet101_pth()
x = paddle.rand([1, 3, 224, 224])
out = resnet18(x)
print(resnet18.fc.weight)'''
'''bottleneck = nn.Linear(resnet18.fc.in_features, 256)
print(bottleneck)'''

# 构造预训练模型：
'''resnet18 = models.resnet18_pth(pretrained=True)
x = paddle.rand([1, 3, 224, 224])
out = resnet18(x)
print(out)
'''
from models.basenet import *
import paddle
from x2paddle import models
from paddle.vision.models import resnet101
import paddle.nn as nn
from x2paddle import torch2paddle

'''x = paddle.rand([1, 3, 224, 224])
out = model(x)
out.backward()
print(out[0][1:10])
out = model(x)
out.backward()
print(out[0][1:10])'''

'''G = ResBottle('resnet101')
data = paddle.rand([64,3,224,224])
out1 = G(data)
out1.backward()
out1 = G(data)
out1.backward()'''

res1 = resnet101()
res2 = models.resnet101_pth()

'''print(list(res1.children()))
print("#############################################################")
print(list(res2.children()))'''

child1 = list(res1.children())
child1.pop()

class testres1(nn.Layer):
    def __init__(self):
        super(testres1, self).__init__()
        self.f1 = nn.Sequential(*child1)
        self.f2 = nn.Linear(2048, 256, weight_attr=nn.initializer.Normal(0, 0.005), bias_attr=nn.initializer.Constant(0.1))
        #torch2paddle.normal_init_(self.f2.weight.data, 0, 0.005)
        #torch2paddle.constant_init_(self.f2.bias.data, 0.1)
        #self.f2.weight.data = paddle.normal(0, 0.005, [2048,256])
        #self.f2.bias.data = nn.initializer.Constant(0.1)
        self.dim = 256

    def forward(self, x):
        x = self.f1(x)
        x = x.view(x.size(0), -1)
        x = self.f2(x)
        x = x.view(x.size(0), self.dim)
        return x


test1 = testres1()
print(test1.f2.weight.data)
data = paddle.rand([64,3,224,224])
out1 = test1(data)
print(out1)
out1.backward()
out1 = test1(data)
out1.backward()
print(out1)