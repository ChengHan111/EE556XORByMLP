import torch
import torch.nn.functional as F
import numpy as np


class MSE(object):
    def __str__(self):
        return 'MSE'

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def loss(self, y, y_pred):
        return 0.5 * np.sum((y_pred - y) ** 2, axis=-1)

    def grad(self, y, y_pred):
        return y_pred - y


class CrossEntropy(object):
    def __init__(self):
        self.eps = np.finfo(float).eps

    def __str__(self):
        return 'CrossEntropy'

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def loss(self, y, y_pred):
        loss = -np.sum(y * np.log(y_pred + self.eps), axis=-1)
        return loss

    def grad(self, y, y_pred):
        grad = y_pred - y
        return grad

def run_mse_fun():
    y_pred = np.array([
        np.array([0.3, 0.4, 0.3]),
        np.array([0.1, 0.1, 0.8])
    ])
    y = np.array([
        np.array([1., 2., 3.]),
        np.array([4., 5., 6.])
    ])
    print("*" * 10, '自定义 mse(L2)', "*" * 10)

    loss_fn = MSE()

    print(loss_fn, 'loss:', loss_fn(y=y, y_pred=y_pred).sum())
    print(loss_fn, 'grad:\n', loss_fn.grad(y=y, y_pred=y_pred))

    """
    pytorch

    因为F.mse_loss中
    f(x)= (y_pred - y)^2
    f'(x) = 2(y_pred - y)
    所以结果会是我们自定义的结果的两背
    """
    y_pred_pytorch = torch.autograd.Variable(torch.FloatTensor(y_pred), requires_grad=True)
    y = torch.FloatTensor(y)

    loss_mse = F.mse_loss(y_pred_pytorch, y, size_average=False)
    print(loss_mse)
    loss_mse.backward()
    grad = y_pred_pytorch.grad.numpy()
    print('pytorch mse grad: \n', grad)


def run_cross_entropy_fun():
    """test cross entropy """
    """
    自定义模型结果
    """
    def softmax(X):
        e_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return e_X / e_X.sum(axis=1, keepdims=True)

    y_before_softmax = np.array([
        np.array([0.3, 0.4, 0.3]),
        np.array([0.1, 0.1, 0.8])
    ])
    y = np.array([
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ])

    print("*" * 10, '自定义cross entropy', "*" * 10)
    y_pred = softmax(y_before_softmax)
    loss_fn = CrossEntropy()

    print(loss_fn, 'loss:', loss_fn(y=y, y_pred=y_pred).sum())
    print(loss_fn, 'grad:\n', loss_fn.grad(y=y, y_pred=y_pred))

    """
    pytorch结果
    """
    """
    1. cross entropy
    """
    print("*" * 10, ' pytorch cross entropy', "*" * 10)
    y_pred_pytorch = torch.autograd.Variable(torch.FloatTensor(y_before_softmax), requires_grad=True)
    y = torch.LongTensor(y.argmax(axis=1))
    loss_cross_entropy = F.cross_entropy(y_pred_pytorch, y, size_average=False).sum()
    print('pytorch cross entropy loss:',loss_cross_entropy)
    loss_cross_entropy.backward()
    grad = y_pred_pytorch.grad.numpy()
    print('pytorch cross entropy grad: \n', grad)

    """
    2. null loss
    """
    print("*" * 10, ' pytorch null loss', "*" * 10)
    y_pred_nll = torch.autograd.Variable(torch.FloatTensor(y_pred), requires_grad=True)
    loss_cross_nll = F.nll_loss(y_pred_nll, y, size_average=False).sum()
    print('pytorch nll loss:', loss_cross_nll)
    loss_cross_nll.backward()
    grad = y_pred_pytorch.grad.numpy()
    print('pytorch null grad: \n', grad)

run_mse_fun()
"""
运行结果：
MSE loss: 38.300000000000004
MSE grad:
 [[-0.7 -1.6 -2.7]
 [-3.9 -4.9 -5.2]]
tensor(76.6000, grad_fn=<MseLossBackward>)
pytorch mse grad:
 [[ -1.4  -3.2  -5.4]
 [ -7.8  -9.8 -10.4]]
********** 自定义cross entropy **********
CrossEntropy loss: 1.7227954009197912
CrossEntropy grad:
 [[ 0.32204346 -0.64408693  0.32204346]
 [ 0.2491434   0.2491434  -0.4982868 ]]
**********  pytorch cross entropy **********
pytorch cross entropy loss: tensor(1.7228, grad_fn=<SumBackward0>)
pytorch cross entropy grad:
 [[ 0.32204345 -0.64408696  0.32204345]
 [ 0.2491434   0.2491434  -0.49828678]]
**********  pytorch null loss **********
pytorch nll loss: tensor(-0.8576, grad_fn=<SumBackward0>)
pytorch null grad:
 [[ 0.32204345 -0.64408696  0.32204345]
 [ 0.2491434   0.2491434  -0.49828678]]
"""