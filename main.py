import json
import os

import pandas as pd
import numpy as np
from utils.prophet_predict import predict
import matplotlib.pyplot as plt

# 讓他可以打中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

if __name__ == "__main__":
    dataset = []
    train_set = []

    # open json file
    with open('data/gold_price.json') as json_file:
        data = json.load(json_file)
        for d, p in data.items():
            dataset.append((d, p))

    trainset = dataset[:-4]
    # test_set = dataset[-26:]

    # data transform pd
    dataset_pd = pd.DataFrame(dataset, columns=["ds", "y"])
    dataset_pd["ds"] = pd.to_datetime(dataset_pd["ds"], format="%Y/%m/%d")

    train_pd = pd.DataFrame(trainset, columns=["ds", "y"])
    train_pd["ds"] = pd.to_datetime(train_pd["ds"], format="%Y/%m/%d")
    #
    # normalization
    dataset_pd["y"] = np.log(dataset_pd["y"])
    train_pd["y"] = np.log(train_pd["y"])

    # predict
    result = predict(dataset=train_pd, future_week_length=4, freq="W-SAT", include_history=True)

    # 繪製結果
    plt.plot(dataset_pd["ds"], round(np.exp(dataset_pd["y"]), 0), '-', color=(0.5, 0.7, 0), label='實際價格')
    plt.plot(result["ds"], round(np.exp(result["yhat"]), 0), '--', color=(0.1, 0.1, 0.1), label='預測價格')
    plt.legend()
    plt.grid(True)
    #
    plt.title("{}-(4周)預測結果".format('黃金現貨 中信局售出'))  # title
    plt.ylabel("價格")  # y label
    plt.xlabel("日期")  # x label
    # plt.show()

    if not os.path.exists('image'):
        os.mkdir('image')
    plt.savefig('image/gold.png')
    plt.close()
