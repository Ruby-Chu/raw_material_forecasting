import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Sequential
import numpy as np
from keras.models import Model

if __name__ == "__main__":
    # define model, parameter set default
    model = Sequential()
    model.add(LSTM(1, input_shape=(3, 1)))

    # test
    # [0.1 0.2 0.3] -> [[[0.1] [0.2] [0.3]]]
    data = np.array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # predict: only output output(y)
    print(model.predict(data))

    #######################################################

    # define model, parameter: return_sequences=True
    model = Sequential()
    model.add(LSTM(1, input_shape=(3, 1), return_sequences=True))

    # test
    # [0.1 0.2 0.3] -> [[[0.1] [0.2] [0.3]]]
    data = np.array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # predict: every node have output(y)
    print(model.predict(data))

    #######################################################

    # define model, parameter: add return_state = True
    inputs1 = Input(shape=(3, 1))
    lstm1 = LSTM(1, return_state=True)(inputs1)
    model = Model(inputs=inputs1, outputs=lstm1)

    # test
    # [0.1 0.2 0.3] -> [[[0.1] [0.2] [0.3]]]
    data = np.array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # predict: retrun hidden state_h and cell state_c
    print(model.predict(data))

    #######################################################

    # define model, parameter: return_sequences=True and return_state = True
    inputs1 = Input(shape=(3, 1))
    lstm1 = LSTM(1, return_sequences=True)(inputs1)
    lstm2 = LSTM(1, return_sequences=True, return_state=True)(lstm1)
    model = Model(inputs=inputs1, outputs=lstm2)

    # test
    # [0.1 0.2 0.3] -> [[[0.1] [0.2] [0.3]]]
    data = np.array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # predict: retrun output(y), state_h, state_c
    print(model.predict(data))
