# 原物料價格預測

``` mermaid
flowchart LR;
    1[load data] --> 2[transform] --> 3[split] --> 4[feature normalization] --> 5[train model and evaluate] --> 6[prediction] --> 7[plot] & 8[RMSE/MAPE];
    5[train model and evaluate] --check--> 3[split];
```

## 黃金行情走勢(美元-盎司) look_back=1
![image info](./image/gold.png)

## 鋼筋豐興廠交價(元-噸) look_back=1
![image info](./image/steel.png)

## 廢鋼-豐興(元-噸) look_back=1
![image info](./image/scrap_steel.png)