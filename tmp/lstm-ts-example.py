# multivariate one step problem with lstm
import tensorflow as tf
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])

# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
target = in_seq2
# define generator
n_features = dataset.shape[1]
window_size = 2
n_output = target.shape[1]
generator = TimeseriesGenerator(dataset, target, length=window_size, batch_size=3)

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(window_size, n_features)))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=300, verbose=1)

# make a one step prediction out of sample
x_input = array([[90, 95], [100, 105]]).reshape((1, window_size, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
