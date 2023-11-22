#
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import os

# 可以打中文字
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']


# 以前的資料:x, 當下資料:y
def create_dateset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        feature = dataset[i:(i + look_back), 0]  # t-n,....,t-1
        dataX.append(feature)
        dataY.append(dataset[i + look_back])  # t
    return np.array(dataX), np.array(dataY)


if __name__ == "__main__":
    infos = []

    dicts = {
        "gold": "黃金行情走勢(美元-盎司)",
        "steel": "鋼筋豐興廠交價(元-噸)",
        "scrap_steel": "廢鋼-豐興(元-噸)"

    }

    # filename = '廢鋼-豐興(元-噸)'

    for key, filename in dicts.items():
        # get information
        with open("json/{}.json".format(filename)) as fid:
            data = json.load(fid)

            for dt, price in data.items():
                infos.append((dt, price))

        df = pd.DataFrame(infos, columns=["dt", "price"])

        # plot
        df2 = df.set_index("dt")
        # df2.plot(legend=None)
        # plt.xticks(rotation=30)
        # plt.show()

        # transform
        dataset = df2.values
        dataset = dataset.astype('float32')

        # X 常態化
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # 資料分割
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # # 以前前資料為x，當期資料為Y
        look_back = 3
        trainX, trainY = create_dateset(train, look_back)
        testX, testY = create_dateset(test, look_back)

        # [筆數、落後期數、dim]
        # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # [筆數、落後期數、x dim]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # 建立模型
        # model = Sequential()
        # model.add(LSTM(4, input_shape=(1, look_back)))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)

        model = Sequential()
        model.add(LSTM(4, input_shape=(look_back, 1)))
        batch_size = 1

        # Stacked LSTM
        model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
        model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)
        for i in range(100):
            model.fit(trainX, trainY, epochs=1, batch_size=batch_size)
            # 重置狀態(cell state)
            model.reset_states()

        # 模型評估
        trainPredict = model.predict(trainX, batch_size=batch_size)
        # 重置狀態(cell state)
        model.reset_states()
        testPredict = model.predict(testX, batch_size=batch_size)

        # 還原
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform(trainY)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform(testY)

        # 計算 RMSE
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
        s1 = 'Train Score: %.2f RMSE' % trainScore
        print(s1)
        testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
        s2 = 'Test Score: %.2f RMSE' % testScore
        print(s2)

        # 訓練資料的X/Y
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

        # 測試資料X/Y
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

        # 繪圖
        plt.plot(scaler.inverse_transform(dataset), color=(0.5, 0.7, 0), label='實際價格', linewidth=1)
        plt.plot(trainPredictPlot, '--', color=(0.5, 0.7, 0.6), label='預測(訓練集)價格', linewidth=1)
        plt.plot(testPredictPlot, '--', color=(0.5, 0.1, 0.6), label='預測(測試集)價格', linewidth=1)
        # plt.show()
        plt.legend()
        plt.grid(True)
        plt.title("{},{},{}".format(filename, s1, s2))  # title
        plt.ylabel(filename)  # y label

        if not os.path.exists('image'):
            os.mkdir('image')
        # gold, steel, scrap_steel
        plt.savefig('image/{}.png'.format(key))

        plt.close()
