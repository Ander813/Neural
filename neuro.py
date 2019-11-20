import numpy as np

#функция для удобной генерации весов
#in_ сколько нейронов на слое из которого выходят синопсы
#out сколько нейронов на слое, куда ведут синопсы
def generate_weights(in_, out):
    return np.random.sample((in_, out))

#функция активатор. в данном случае сигмойд.
def activator(x, der=False):
    if der:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

#на вход подаем слой и веса, которые ведут к следующему слою
#пример l3 = l2, weights2
def forward(layer, weight):
    return activator(np.dot(layer, weight))

#поиск ошибки. На вход подаем слой для которого ищем ошибку, веса ведущие к этому слою и предыдущую ошибку
#пример l2_error = l2, weight2, l3_error
def find_error(layer, weight, error):
    return np.dot(weight, error)*activator(layer, True)

#поправка весов с учетом ошибки. На вход подаем слой из которого исходят синапсы,
#веса, для которых ищем поправку, коэф обучаемости, и ошибку слоя в который ведут веса.
#пример weights2 = l2, weights2, коэф[0;1], l3_error
def adjastment(layer, weight, koef, error):
    for i in range(weight.shape[0]):
        weight[i] += koef*layer[i]*error
    return weight


class NeuralNetwork:
    def __init__(self, exp_res, offsets, weights, start=None):
        #генерируем веса из словаря
        self.weights = [generate_weights(j[0], j[1]) for i,j in weights.items()]
        #определяем размер тренировочного набора
        self.LEN = start.shape[0]
        #определяем последний слой (для последнего слоя принцып обработки немного отличается)
        self.LAST = len(weights)+1
        #создаем словарь в котором будут храниться все слои
        self.layers = {}
        #слои где должы быть добавлены нейроны смещения
        self.offsets = offsets
        #ошибки слоев
        self.errors = {}
        #ожидаемый результат
        self.exp_res = exp_res
        #проверяем нужно ли добавить нейрон смещения для 1 слоя
        if self.offsets['l1']:
            self.start = np.array([np.append(i,1) for i in start])
        else:
            self.start = start

    #считаем слои друг за другом
    def update_layers(self):
        for i in range(2, len(self.weights)+2):
            #слой для которого считаем         предыдущий слой      веса ведущие в этот слой
            self.layers['l'+str(i)] = forward(self.layers['l'+str(i-1)], self.weights[i-2])\
                if self.offsets['l' + str(i)] == False else \
                np.append(forward(self.layers['l'+str(i-1)], self.weights[i-2]), 1)

    #Вычисляем ошибку
    def calc_error(self, num):
        #ошибка последнего слоя = (ожидаемый результат - полученный) * производную последнего слоя
        self.errors['l'+str(self.LAST)+'_error'] = (self.exp_res[num]-self.layers['l'+str(self.LAST)]) * \
            activator(self.layers['l'+str(self.LAST)], True)
        for i in range(len(self.layers)-1,1,-1):
            if self.offsets['l'+str(i)]:
                self.errors['l'+str(i)+'_error'] = find_error(self.layers['l' + str(i)][:-1], self.weights[i-1][:-1],\
                    self.errors['l'+str(i+1)+'_error'])
            else:
                self.errors['l'+str(i)+'_error'] = find_error(self.layers['l' + str(i)], self.weights[i-1],\
                    self.errors['l'+str(i+1)+'_error'])

    #обновляем веса
    def update_weights(self, koef):
        for i in range(len(self.weights)):
            self.weights[i] = adjastment(self.layers['l'+ str(i+1)], self.weights[i], koef, self.errors['l' + str(i+2) + '_error'])

    #функция отвечающая за обучение нейронной сети.
    #на вход получает коэф обучаемости, количество эпох и через какое количество эпох выводить результат обучения сети
    #если control == None результат не печатается, а веса возвращаются как результат работы функции.
    def run(self, koef, epoch, control=None):
        for i in range(epoch+1):
            total_error = 0
            for n in range(self.LEN):
                self.layers['l1'] = self.start[n]
                self.update_layers()
                self.calc_error(n)
                self.update_weights(koef)
                if i % control == 0 and control is not None:
                    for j in self.errors['l'+str(self.LAST)+'_error']:
                        total_error += (1/len(self.layers['l'+str(self.LAST)]))*j*j
                    print(f'ожидаемый результат\t{self.exp_res[n]}')
                    print(f'полученный\t{self.layers["l"+str(self.LAST)]}')
                    print(f'итерация\t{i}')
                    print('\n\n')
            if i % control == 0 and control is not None:
                print(f'ошибка\t{total_error}')
                print('\n\n')

        if control is None:
            return self.weights

    def test(self, test):
        if type(test) != 'numpy.ndarray':
            test = np.array(test)

        if len(test.shape) == 1:
            if self.offsets['l1']:
                self.layers['l1'] = np.append(test, 1)
            else:
                self.layers['l1'] = test
        else:
            if self.offsets['l1']:
                self.layers['l1'] = np.array([np.append(i,1) for i in test])
            else:
                self.layers['l1'] = test

        self.update_layers()
        return self.layers['l' + str(self.LAST)]


if __name__ == '__main__':
    test= np.array([[0.1, 0.1],
              [0.2, 0.1],
              [0.2, 0.2],
              [0.3, 0.1],
              [0.3, 0.2],
              [0.3, 0.3],
              [0.4, 0.1],
              [0.4, 0.2],
              [0.4, 0.3],
              [0.4, 0.4],
              [0.5, 0.1],
              [0.5, 0.2],
              [0.5, 0.3],
              [0.5, 0.4],
              [0.6, 0.1],
              [0.6, 0.2],
              [0.6, 0.3],
              [0.7, 0.1],
              [0.7, 0.2],
              [0.8, 0.1]])
    exp_res = np.array([0.1,0.31,0.4,0.42,0.51,0.6,0.53,0.62,
                        0.71,0.8,0.64,0.73,0.82,0.91,0.75,0.84,
                        0.93, 0.86,0.95,0.97]).T

    #слои на которых нужно добавить нейроны смещения.
    offsets = {'l1': True,
               'l2': True,
               'l3': True,
               'l4': False}

    #определяем сколько на каждом слое будет нейронов.
    #разница 2 и 3 получается из-за того что есть нейрон смещения.
    weights = {'l1': (3,3),
               'l2': (4,3),
               'l3': (4,1)}


    training = NeuralNetwork(exp_res, offsets, weights, start=test)
    training.run(0.5, 10000, 2000)
