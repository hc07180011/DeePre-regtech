import time
import json
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt


target = 2499
site = 'https://query1.finance.yahoo.com/v8/finance/chart/{}.TW?period1=0&period2={}&interval=1d&events=history&=hP2rOschxO0'.format(target, int(time.time()))

response = requests.get(site)
tickers = json.loads(response.text)['chart']['result'][0]['indicators']['quote'][0]
tickers['timestamp'] = json.loads(response.text)['chart']['result'][0]['timestamp']


tracking_years = np.arange(2000, 2021, 1)
ticker = [[1e-9,] for _ in range(len(tracking_years))]
volume = [[1e-9,] for _ in range(len(tracking_years))]

for time_, close_, volume_ in zip(tickers['timestamp'], tickers['close'], tickers['volume']):
    idx = int(str(datetime.datetime.fromtimestamp(time_)).split('-')[0]) - tracking_years[0]
    ticker[idx].append(close_ if close_ else 0)
    volume[idx].append(volume_ / 1e9 if volume_ else 0)

ticker = [np.mean(x) for x in ticker]
volume = [np.mean(x) for x in volume]


fig, ax = plt.subplots()
ax.plot(ticker)
#ax.plot(volume)
ax.set_xticks(np.arange(0, 21, 1).tolist())
ax.set_xticklabels(np.arange(2000, 2021, 1).tolist(), fontsize=4)
plt.savefig('2499_stock_history.png')
plt.close()
