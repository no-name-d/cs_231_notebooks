from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, weights_return=False):
        """
        Обучение линейного классификатора с помощью стохастического градиентного спуска.

        Inputs:
        - X: Массив обучающих данных numpy размером (N, D), содержащий N
             обучающих образцов размерностью D.
        - y: Массив обучающих меток numpy размером (N,) ; y[i] = c
             означает, что образец X[i] имеет метку 0 <= c < C для C классов.
        - learning_rate: (float) скорость обучения для оптимизации.
        - reg: (float) сила регуляризации.
        - num_iters: (integer) число шагов оптимизации.
        - batch_size: (integer) число обучающих примеров для каждого шага.
        - verbose: (boolean) Если true, выводит прогресс в процессе оптимизации.

        Outputs:
        Список, содержащий значение функции потерь на каждой итерации обучения.
        """
        num_train, dim = X.shape
        # print(X.shape)
        num_classes = np.max(y) + 1 # предположим, y принимает значения 0...K-1 где K - число классов
        if self.W is None:
            # инициализируем веса
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Запускаем SGD, чтобы оптимизировать W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Выберите batch_size (размер пакета) элементов из обучающих данных и   #
            # соответствующих им меток, чтобы использовать их для градиентного      #
            # спуска. Сохраните данные в X_batch, а их метки в y_batch. X_batch     #
            # должен иметь размер (batch_size, dim), а y_batch - (batch_size,).     #                       #
            #                                                                       #
            # Подсказка: Используйте np.random.choice                               #
            #########################################################################
            # *****START OF YOUR CODE*****

            X_b = np.split(X, batch_size)
            y_b = np.split(y, batch_size)
            batch_num = np.random.choice([x for x in range(batch_size)])
            # print(batch_num)
            X_batch = X_b[batch_num]
            y_batch = y_b[batch_num]
            pass

            # *****END OF YOUR CODE*****

            # Оцениваем потери и градиент

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Выполняем обновление параметров
            #########################################################################
            # TODO:                                                                 #
            # Обновите веса, используя градиент и скорость обучения.                #
            #########################################################################
            # *****START OF YOUR CODE*****

            self.W -= grad * learning_rate
            pass

            # *****END OF YOUR CODE*****

            if verbose and it % 10 == 0:
                print('Итерация %d из %d: потери - %f' % (it, num_iters, loss))
        
        # print(loss_history)
        ret = np.dot(X, self.W)
        clasess = np.argmax(ret, axis=1)
        # print(clasess.shape, clasess[:100])
        y_pred = clasess

        if weights_return == True:
            return loss_history, y_pred, self.W
        return loss_history, y_pred


    def predict(self, X):
        """
        Использование полученных весов для прогнозирования меток. 

        Inputs:
        - X: Массив обучающих данных numpy размером (N, D), содержащий N
             обучающих образцов размером D.

        Returns:
        - y_pred: Спрогнозированные метки для данных X. y_pred - одномерный массив 
                  размера N, содержащий прогнозы для меток классов.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Реализуйте метод и сохраните метки в y_pred.                            #
        ###########################################################################
        # *****START OF YOUR CODE*****

        # _, two = self.loss(X_batch, y_batch, reg)
        # print(X.shape)
        ret = np.dot(X, self.W)
        # print(ret.shape)
        clasess = np.argmax(ret, axis=1)
        # print(clasess.shape, clasess[:100])
        y_pred = clasess

        pass


        # *****END OF YOUR CODE*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Вычисление функции потерь и ей производной.
        Функция будет переопределена в подклассах.

        Inputs:
        - X_batch: Массив numpy размером (N, D), содержащий минипакет из N
                   точек данных; каждая точка имеет размерность D.
        - y_batch: Массив numpy с метками для минипакета размерностью (N,).
        - reg: (float) сила регуляризации.

        Returns: Кортеж, содержащий:
        - потери (число)
        - градиент по отношению к self.W (массив той же размерности, что и W)
        """
        
        return svm_loss_naive(self.W, X_batch, y_batch, reg)
        pass

    def best_svm(self, W, X):
        y_pred = np.zeros(X.shape[0])

        ret = np.dot(X, W)
        # print(ret.shape)
        clasess = np.argmax(ret, axis=1)
        # print(clasess.shape, clasess[:100])
        y_pred = clasess
        return y_pred

class LinearSVM(LinearClassifier):
    """ Подкласс, использующий функцию потерь мультиклассового SVM """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ Подкласс, использующий функцию потерь Softmax + Cross-entropy """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
