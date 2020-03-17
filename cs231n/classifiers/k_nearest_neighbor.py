from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ Классификатор kNN """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Обучение классификатора. 

        Inputs:
        - X: Массив обучающих данных размером (num_train, D), содержащий
             num_train образцов размерности D.
        - y: Массив numpy размером (N,) содержащий обучающие метки, где
          y[i] - метка для X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Прогнозирование меток для тестовых даннх.

        Inputs:
        - X: Массив тестовых данных размером (num_train, D) содержащий
             num_train образцов размерности D.
        - k: Число ближайших соседей, которые будут голосовать за метки.
        - num_loops: Определяет, какую реализацию использовать для вычисления расстояний между
                     обучающими и тестовыми точками.

        Returns:
        - y: Массив numpy размером (num_test,) содержащий прогнозы меток для тестовых данных,
             где y[i] это прогноз метки для точки X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Неверное значение %d для num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Вычисление расстояния между каждой тестовой точкой в X и каждой обучающей точкой
        в self.X_train с помощью вложенного цикла.

        Inputs:
        - X: Массив тестовых данных.

        Returns:
        - dists: Массив numpy размером (num_test, num_train) где dists[i, j] -
                 это евклидово расстояние между i-й тестовой точкой и j-1 обучающей точкой.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Вычислите расстояние L2 между i-й тестовой точкой и j-й обучающей #
                # точкой, а затем сохраните результат в dists[i, j]. Постарайтесь   #
                # не использовать дополнительные циклы и функцию np.linalg.norm().  #
                #####################################################################
                # *****START OF YOUR CODE*****
                dists[i, j] = (np.sum((X[i] - self.X_train[j]) ** 2)) ** 0.5
                pass

                # *****END OF YOUR CODE*****
        print(X.shape, ' Массив тестовых данных', X[0])
        # print (dists[228, 2228])
        # print('class matrix: ', type(X), type(self.X_train))
        # print(self.X_train.shape, 'Maccив тренировочных данных', self.X_train[0])
        return dists

    def compute_distances_one_loop(self, X):
        """
        Вычисление расстояния между каждой тестовой точкой в X и каждой обучающей точкой
        в self.X_train с помощью одного цикла.

        Input / Output: аналогичные предыдущей функции
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Вычислите расстояние L2 между i-й тестовой точкой и всеми           #
            # обучающими точками, а затем сохраните результат в dists[i, :].      #
            # Не используйте np.linalg.norm().                                    #
            #######################################################################
            # *****START OF YOUR CODE*****
            x, y = X[i], self.X_train

            x2 = np.sum(x**2, axis=0, keepdims=True)
            y2 = np.sum(y**2, axis=1)
            xy = np.dot(x, y.T)
            dists[i, :] = np.sqrt(x2 - 2*xy + y2)
            # Понять, почему оно работает
            if i == 2:
                print(x2, y2)
                print(xy, xy.shape, y.T.shape)
            # *****END OF YOUR CODE*****
        print(num_test, num_train, dists.shape, X.shape, self.X_train.shape)
        return dists

    def compute_distances_no_loops(self, X):
        """
        Вычисление расстояния между каждой тестовой точкой в X и каждой обучающей точкой
        в self.X_train без использзования циклов.

        Input / Output: аналогичные
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Вычислите расстояние L2 между всеми тестовыми и обучающими точками    #
        # без использования циклов; результат сохраните в dists.                #
        #                                                                       #
        # Реализуйте эту функцию, используя только базовые операции с массивами #
        # без функций scipy и np.linalg.norm().                                 #
        #                                                                       #
        # Подсказка: Попробуйте использовать умножение матриц                   #
        #########################################################################
        # *****START OF YOUR CODE*****
        x, y = X, self.X_train

        x2 = np.sum(x**2, axis=1, keepdims=True)
        y2 = np.sum(y**2, axis=1)
        xy = np.dot(x, y.T)
        dists = np.sqrt(x2 - 2 * xy + y2)
        pass

        # *****END OF YOUR CODE*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Предсказание меток с помощью матрицы расстояний между тестовыми и обучающими точками.

        Inputs:
        - dists: Массив numpy array размером (num_test, num_train) где dists[i, j] -
                 это расстояние между i-й тестовой точкой и j-й обучающей точкой.

        Returns:
        - y: Массив numpy размером (num_test,) содержащий прогноз меток для тестовых данных,
             где y[i] это прогноз метки для точки X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        new = []  # test
        for i in range(num_test):
            # Список длиной k содержит метки k ближайших соседей
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Используйте матрицу расстояний, чтобы найти k ближайших соседей для   #
            # i-й тестовой точки, затем используйте self.y_train и найдите метки    #
            # этих соседей. Сохраните метки в closest_y.                           #
            # Подсказка: Посмотрите на функцию numpy.argsort.                       #
            #########################################################################
            # *****START OF YOUR CODE*****
            
            py_list = list(dists[i])
            for num in range(k):
                min_index = py_list.index(min(py_list))  # n
                py_list = py_list[0:min_index] + py_list[min_index+1:]   # n
                closest_y.append(self.y_train[min_index]) 
            count_list = []
            for j in range(len(closest_y)):
                count_list.append(closest_y.count(closest_y[j]))
            moda = closest_y[count_list.index(max(count_list))]
            # if i == 2:
                # print(moda, count_list)




            # new_sort = np.argsort(dists[i])
            # for j in range(len(dists[i])):
            #     if dists[i][j] == new_sort[:k+1]:
            #         closest_y.append(j)
                
            
            pass

            # *****END OF YOUR CODE*****
            #########################################################################
            # TODO:                                                                 #
            # Теперь вам необходимо найти наиболее подходящую (часто встречающуюся) #
            # метку в списке closest_y.                                             #
            # Сохраните её в y_pred[i].                                             #
            #########################################################################
            # *****START OF YOUR CODE*****
            y_pred[i] = moda
            if i == 499:
                # print(y_pred)
                pass

            # *****END OF YOUR CODE*****

        # print(num_test)
        # print(dists[0, 0], dists[0, 1], dists[0, 2])
        # print(len(new), new[0:10])
        return y_pred
