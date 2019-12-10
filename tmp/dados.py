import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

df = pd.read_csv("https://raw.githubusercontent.com/lmlima/PoolAttendance/master/data/ospa-completo.csv?token=ABOSJ7YLKCQ6FEDFKLX4GUK5ZRPS6", parse_dates=["Date"])

df["weekday"] = df["Date"].dt.dayofyear
# df[df_few_orig.Pool == "Adelphi"][["Attendance"]].plot()
# df[(df.weekday == 4) & (df.Pool == "Astoria")][["Attendance"]].plot()

# plt.show()
series = df[df.Pool == "Adelphi"][["Attendance"]].reset_index(drop=True)
split_time = 100
# time_valid = [split_time:]

x_valid = series[split_time:]
naive_forecast = series[split_time - 1:-1]
naive_forecast.index += 1

result = pd.concat([x_valid, naive_forecast], axis=1, sort=False)
result.plot(style=['-', '--'])
plt.show()

x_valid = x_valid.to_numpy()
naive_forecast = naive_forecast.to_numpy()



x_valid = x_valid.reshape((len(x_valid),))
naive_forecast = naive_forecast.reshape((len(naive_forecast),))

print(tf.keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

