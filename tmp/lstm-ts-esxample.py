# univariate one step problem with mlp
from numpy import array

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# define generator
n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=8)

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=300, verbose=1)

# make a one step prediction out of sample
x_input = array([9, 10]).reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)

print(yhat)
