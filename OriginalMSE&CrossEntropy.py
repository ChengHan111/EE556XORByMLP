import numpy as np


class MSE(object):
    """
    L2
    SquaredError
    最小均方误差
    回归结果,神经元输出后结果计算损失
    """
    def __str__(self):
        return 'MSE'

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def loss(self, y, y_pred):
        """
        loss = 1/2 * (y - y_pred)^2
        :param y:class:`ndarray <numpy.ndarray>` 样本结果(n, m)
        :param y_pred:class:`ndarray <numpy.ndarray>` 样本预测结果(n, m)
        :return: shape(n, m)
        """
        return 0.5 * np.sum((y_pred - y) ** 2, axis=-1)

    def grad(self, y, y_pred):
        """
        一阶导数： y_pred - y
        :param y: class:`ndarray <numpy.ndarray>`样本结果(n, m)
        :param y_pred:class:`ndarray <numpy.ndarray>` 样本预测结果(n, m)
        :return: shape(n, m)
        """
        return y_pred - y


class CrossEntropy(object):
    def __init__(self):
        self.eps = np.finfo(float).eps

    def __str__(self):
        return 'CrossEntropy'

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def loss(self, y, y_pred):
        """
        loss = - sum_x p(x) log q(x)
        :param y:class:`ndarray <numpy.ndarray>` 样本结果(n, m)
        :param y_pred:class:`ndarray <numpy.ndarray>` 样本预测结果(n, m)
        :return: shape(n, m)
        """
        loss = -np.sum(y * np.log(y_pred + self.eps), axis=-1)
        return loss

    def grad(self, y, y_pred):
        """
        这儿的一阶导数包括了softmax部分
        :param y:class:`ndarray <numpy.ndarray>` 样本结果(n, m)
        :param y_pred:class:`ndarray <numpy.ndarray>` 样本预测结果(n, m)
        :return: shape(n, m)
        """
        grad = y_pred - y
        return grad