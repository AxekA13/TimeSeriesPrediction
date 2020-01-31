### Только для стационарных датасетов"
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=True, world_readable=True)

data = pd.read_csv(r'C:\Users\nasty\PycharmProjects\OxaProject\PJME_hourly.csv', index_col=0)
data.head()
data.index = pd.to_datetime(data.index)
data = data.sort_values('Datetime')
data = data[140000:]


print(data)

cutoff_point = int(len(data) * 0.9)

train = data[:cutoff_point]
test = data[cutoff_point + 1:]

from pmdarima.arima import auto_arima

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=False)
print(stepwise_model.aic())

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=test.shape[0])
print(future_forecast)
future_forecast = pd.DataFrame(future_forecast, index=test.index)
plt.plot(future_forecast,label='Prediction')
plt.plot(test,label='True')
plt.legend()
plt.show()

