from prophet import Prophet


def predict(dataset, holidays=None, future_week_length=26, freq="W-MON", yearly_seasonality=True,
            weekly_seasonality=True, daily_seasonality=False):
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        holidays=holidays)

    model.fit(dataset)

    # 預測歷史資料
    # future_dataframe = model.make_future_dataframe(periods=future_week_length, freq=freq, include_history=True)
    future_dataframe = model.make_future_dataframe(periods=future_week_length, freq=freq, include_history=False)
    result = model.predict(future_dataframe)

    info = result.loc[:, ["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return info