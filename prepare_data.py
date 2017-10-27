import numpy as np


def data(data_file='train_data.csv'):
    f = open(data_file)
    inputs = []
    outputs = []
    for line in f.readlines():
        line = [float(x) for x in line.strip().split(',')]
        input = line[0:-1]
        temp = []
        for i in range(0, len(input), 2):
            temp.append(input[i:i+2])
        inputs.append(temp)
        outputs.append(line[-1])
    return np.array(inputs), outputs
