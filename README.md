# 原物料價格預測

``` mermaid
flowchart LR;
	1[載入資料] --> 2[資料轉換] --> 3[資料切割] --> 4[feature normalization] --> 5[訓練模型與評估] --> 6[prediction] --> 7[繪製實際及預測資料的圖形] --> 8[效能評估RMSE/MAPE];
	5[訓練模型與評估] --測試驗證參數--> 3[資料切割];
```