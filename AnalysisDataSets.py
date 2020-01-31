import pandas as pd
import matplotlib.pyplot as plt
# Стандартные импорты plotly
import plotly.graph_objs as go
from plotly.offline import plot_mpl, iplot
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.offline import iplot
import cufflinks
from statsmodels.tsa.stattools import adfuller

cufflinks.go_offline()
# Устанавливаем глобальную тему
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)

df = pd.read_csv(r'C:\Users\nasty\PycharmProjects\OxaProject\PJME_hourly.csv', index_col=0, parse_dates=True)

# Графическое изображение графика даты и значения
def test_stationary(timeseries, window=12):
    # Rolling statistics
    movingAverage = timeseries.rolling(window=window).mean()
    movingSTD = timeseries.rolling(window=window).std()

    # Dickey Fuller test
    print('Results of Dickey Fuller Test:\n')
    dftest = adfuller(timeseries['PJME_MW'], autolag='AIC')
    print(dftest)
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
df_2 = pd.read_csv(r'C:\Users\nasty\PycharmProjects\OxaProject\cleaned_costs.csv', index_col=0, parse_dates=True)
test_stationary(df,12)
len = len(df_2.columns[2:])
fig = make_subplots(rows=len, cols=1)
temp = 1
for col in df_2.columns[2:]:
    fig.append_trace(go.Scatter(x=df_2['Date'], y=df_2[col], name=col), row=temp, col=1)
    temp = temp + 1
fig.show()


"""
data = df[:1000]
df = df.reset_index()
fig = go.Figure(go.Scatter(x=df['Datetime'], y=df['DUQ_MW']))
fig.show()

# Анализ на сезонность
result = seasonal_decompose(data, model='multiplicative',freq=100)
fig = result.plot()
plot_mpl(fig)
"""
