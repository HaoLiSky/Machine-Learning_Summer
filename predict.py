import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.models import Model
import numpy as np


class ATOMModel:
    def __init__(self, restore, n=55):
        inputs_data = []
        for i in range(n):
            inputs_data.append(Input((2,), ))

        atom_input = Input((2,))
        hidden = Dense(10, activation='sigmoid')(atom_input)
        prediction = Dense(1)(hidden)
        model = Model(inputs=atom_input, outputs=prediction)

        outputs = []
        for input in inputs_data:
            outputs.append(model(input))

        result = keras.layers.add(outputs)

        final_model = Model(inputs_data, result)

        final_model.load_weights(restore)
        self.prediction = final_model
        self.n = n

    def predict(self, data):
        inputs = []
        for i in range(self.n):
            inputs.append(np.array(data[:, i]))
        return self.prediction.predict(inputs)


def load_data(data_file='predict_data.csv'):
    f = open(data_file)
    inputs = []
    for line in f.readlines():
        input = [float(x) for x in line.strip().split(',')]
        temp = []
        for i in range(0, int(len(input)/2)*2, 2):
            temp.append(input[i:i + 2])
        inputs.append(temp)
    return np.array(inputs)


if __name__ == '__main__':
    model = ATOMModel('atom_model', n=55)
    data = load_data('predict_data.csv')
    result = model.predict(data)
    print(result)
   	

