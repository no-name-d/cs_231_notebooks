from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Функция потерь SVM, наивная реализация (с циклами).

    Исходные данные имеют размерность D, всего есть C классов, и мы оперируем минипакетами
    с N примерами.

    Inputs:
    - W: Массив весов размером (D, C).
    - X: Массив с минипакетом данных размером (N, D).
    - y: Массив обучающих меток размером (N,); y[i] = c означает, что
         X[i] имеет метку c, где 0 <= c < C.
    - reg: (float) сила регуляризации

    Returns: Кортеж, содрежащий:
    - потери (число)
    - градиент по отношению к self.W (массив той же размерности, что и W)
    """
    dW = np.zeros(W.shape) # инициализируем градиент нулями

    # вычисляем потери и градиент
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        # if i == 1:
            # print(scores.shape)
        correct_class_score = scores[y[i]]
        # print(correct_class_score)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            # print(margin)
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]
                # stupid regulation
    # print(correct_class_score)

    # Сейчас потери - это сумма по всем обучающим примерам, но мы хотим, чтобы они были
    # усреднённым значением, поэтому делим их на число примеров num_train.
    # print(loss)
    loss /= num_train
    dW /= num_train

    # Добавляем регуляризацию
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Вычислите градиент функции потерь и сохраните его в dW.                   #
    # Вместо того, чтобы сначала вычислять потери, а затем считать производные, #
    # проще будет искать производные и потери одновременно. Вы можете просто    #
    # немного изменить код выше, чтобы посчитать градиент.                      #
    #############################################################################
    # *****START OF YOUR CODE*****

    # print(dW.shape, num_classes, num_train)

    # print(dW[0:100, :])
    # dW = np.sum(dW)
    # print(dW)
    pass

    # *****END OF YOUR CODE*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Функция потерь SVM, векторизованная реализация.

    """
    loss = 0.0
    num_train = X.shape[0]
    dW = np.zeros(W.shape) # инициализируем градиент нулями

    #############################################################################
    # TODO:                                                                     #
    # Реализуйте векторизованную верстю вычисления потерь SVM и сохраните       #
    # результат в переменной loss.                                              #
    #############################################################################
    # *****START OF YOUR CODE*****

    delta = 1.0
    scores_matrix = X.dot(W)
    # print(scores_matrix.shape)
    yi_scores = scores_matrix[np.arange(scores_matrix.shape[0]),y]
    margins = np.maximum(0, scores_matrix - np.matrix(yi_scores).T + delta)
    margins[np.arange(num_train),y] = 0  # Обнуляем оценку для совпадающих классов
    # margins[y] = 0   
    loss = np.mean(np.sum(margins, axis=1))
    loss += 0.5 * reg * np.sum (W * W)
    # loss /= X.shape[0] # cоздаём усредненное
    # loss += reg * np.sum(W * W) # ... и возвращаем нормализованное
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = (-1 * row_sum.T)
    dW = np.dot(X.T, binary)

    # Average
    dW /= num_train

    # Regularize
    dW += reg * W

    # *****END OF YOUR CODE*****

    #############################################################################
    # TODO:                                                                     #
    # Реализуйте векторизованную верстю вычисления градиента SVM и сохраните    #
    # результат в переменной dW.                                                #
    #                                                                           #
    # Подсказка: используйте код, который вы применяли для вычисления потерь.   #
    #############################################################################
    # *****START OF YOUR CODE*****

    pass

    # *****END OF YOUR CODE*****

    return loss, dW
