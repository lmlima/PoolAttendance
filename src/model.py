def lstm_model(data_window, data_features, n_labels, output=True):
    model = tf.keras.Sequential()

    # LSTM Layer
    # model.add(tf.keras.layers.LSTM(32, input_shape=(data_window, data_features), return_sequences=True))
    # model.add(tf.keras.layers.LSTM(128, input_shape=(data_window, data_features)))
    model.add(tf.keras.layers.LSTM(128, input_shape=(data_window, data_features), dropout=0.1, recurrent_dropout=0.1))
    # model.add(tf.keras.layers.Conv1D(128, data_window, input_shape=(data_window, data_features)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    #
    # # DNN layer
    # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.1))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.1))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.1))
    # model.add(tf.keras.layers.BatchNormalization())

    # Output layer
    if output:
        model.add( tf.keras.layers.Dense(units=n_labels, activation='softmax') )
        # model.add( tf.keras.layers.Dense(units=1, activation='sigmoid') )


    return model