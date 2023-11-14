import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# 前置處理函數，取得模型輸入的格式
# look_back：特徵(X)個數，forward_days：目標(y)個數，jump：移動視窗
def processData(data, look_back, forward_days, jump=1):
    dataX, dataY = [], []
    for i in range(0, len(data) - look_back - forward_days + 1, jump):
        dataX.append(data[i:(i + look_back)])
        dataY.append(data[(i + look_back):(i + look_back + forward_days)])
    return np.array(dataX), np.array(dataY)


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, X=0,1,2,3-----99   Y=100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == "__main__":
    dataset = []

    # get information
    with open("json/黃金行情走勢(美元-盎司).json") as fid:
        data = json.load(fid)

        for dt, price in data.items():
            dataset.append((dt, price))
    # print(dataset)

    pd_dataset = pd.DataFrame(dataset, columns=["dt", "price"])
    # print(pd_dataset)

    # log normalization
    # pd_dataset["price"] = np.log(pd_dataset["price"])
    # print(pd_dataset["price"])

    df1 = pd_dataset.reset_index()['price']
    # print(df1)

    # plt.plot(df1)
    # plt.show()

    # normalize 0~1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    # print(df1)

    # splitting dataset into train and test split
    training_size = int(len(df1) * 0.9)
    test_size = len(df1) - training_size

    train_data = df1[0:training_size, :]
    test_data = df1[training_size:len(df1), :]

    # print(f"training size:{len(trainset)}\ntest size:{len(testset)}")
    # reshape into X=t,t+1,t+2..t+99 and Y=t+100
    time_step = 12
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create the Stacked LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(12, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()

# import json
# import os
#
# import pandas as pd
# import numpy as np
# from utils.prophet_predict import predict
# import matplotlib.pyplot as plt
#
# # 讓他可以打中文
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
#
# if __name__ == "__main__":
#     # all data
#     dataset = []
#     # training
#     training = []
#
#     # test
#
#     # open json file
#     with open('json/廢鋼-豐興(元-噸).json') as json_file:
#         data = json.load(json_file)
#         for d, p in data.items():
#             dataset.append((d, p))
#
#     trainset = dataset[:-4]
#     # test_set = dataset[-26:]
#
#     # data transform pd
#     dataset_pd = pd.DataFrame(dataset, columns=["ds", "y"])
#     dataset_pd["ds"] = pd.to_datetime(dataset_pd["ds"], format="%Y/%m/%d")
#
#     train_pd = pd.DataFrame(trainset, columns=["ds", "y"])
#     train_pd["ds"] = pd.to_datetime(train_pd["ds"], format="%Y/%m/%d")
#     #
#     # normalization
#     dataset_pd["y"] = np.log(dataset_pd["y"])
#     train_pd["y"] = np.log(train_pd["y"])
#
#     # predict
#     result = predict(dataset=train_pd, future_week_length=4, freq="W-SAT", include_history=True)
#
#     # 繪製結果
#     plt.plot(dataset_pd["ds"], round(np.exp(dataset_pd["y"]), 0), '-', color=(0.5, 0.7, 0), label='實際價格')
#     plt.plot(result["ds"], round(np.exp(result["yhat"]), 0), '--', color=(0.1, 0.1, 0.1), label='預測價格')
#     plt.legend()
#     plt.grid(True)
#     #
#     plt.title("{}-(4周)預測結果".format('廢鋼-豐興(元/噸)'))  # title
#     plt.ylabel("價格")  # y label
#     plt.xlabel("日期")  # x label
#     # plt.show()
#
#     if not os.path.exists('image'):
#         os.mkdir('image')
#     plt.savefig('image/gold.png')
#     plt.close()
