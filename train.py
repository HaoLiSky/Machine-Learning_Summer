import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
import numpy as np

from prepare_data import data


def train(train_inputs, train_outputs, valid_inputs, valid_outputs, model_file, num_epochs=50000, loss='mean_squared_error', optimizer=None, n=55):
    inputs_data = []
    for i in range(n):
        inputs_data.append(Input((2, ), ))

    atom_input = Input((2, ))
    hidden = Dense(10, activation='sigmoid')(atom_input)
    prediction = Dense(1)(hidden)
    model = Model(inputs=atom_input, outputs=prediction)

    outputs = []
    for input in inputs_data:
        outputs.append(model(input))

    result = keras.layers.add(outputs)

    final_model = Model(inputs_data, result)

    if not optimizer:
        optimizer = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

    final_model.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])

    inputs = []
    v_inputs = []
    for i in range(n):
        inputs.append(train_inputs[:, i])
        v_inputs.append(valid_inputs[:, i])

    final_model.fit(inputs, train_outputs,
                    epochs=num_epochs,
                    validation_data=(v_inputs, valid_outputs),
                    shuffle=True)

    if model_file != None:
        final_model.save(model_file)
    return final_model

if __name__ == '__main__':
    inputs, outputs = data()
    train_inputs, train_outputs = inputs[0:750], outputs[0:750]
    valid_inputs, valid_outputs = inputs[751:1016], outputs[751:1016]
    train(train_inputs, train_outputs, valid_inputs, valid_outputs,
          'atom_model', num_epochs=50000, loss='mean_squared_error', n=55)
