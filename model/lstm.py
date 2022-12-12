import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.conf import train_size_of_sequence, batch_size_of_sequence, activation_function, length_of_sequence, \
    start_value, end_value, steps_on_range, number_of_epochs
from utils.lib import training, testing, get_input


def main():
    function = get_input()
    range_of_data = np.linspace(start_value, end_value, steps_on_range)
    data = [[function(i)] for i in range_of_data]

    train_data_set = data[:int(train_size_of_sequence * len(data))]
    test_data_set = data[int(train_size_of_sequence * len(data)):]
    train_target_set = train_data_set[length_of_sequence:]
    test_range_of_data_set = range_of_data[int(train_size_of_sequence * len(data)):]

    train_set = tf.keras.utils.timeseries_dataset_from_array(
        sequence_length=length_of_sequence,
        targets=train_target_set,
        batch_size=batch_size_of_sequence,
        data=train_data_set,
    )

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(length_of_sequence, 1)),
        tf.keras.layers.LSTM(number_of_epochs, activation=activation_function, return_sequences=False),
        tf.keras.layers.Dense(units=1)
    ])

    lstm_model.compile(loss=tf.losses.Huber(),
                       optimizer=tf.optimizers.Adam(learning_rate=0.01),
                       metrics=[tf.metrics.MeanAbsoluteError()])

    lstm_model.build()
    lstm_model.summary()

    training(lstm_model, train_set, number_of_epochs)
    forecast = testing(lstm_model, test_data_set)

    plt.figure(figsize=(10, 5))
    plt.title("Forecast", fontsize=6)
    plt.plot(test_range_of_data_set, test_data_set)
    plt.plot(test_range_of_data_set, forecast)
    plt.show()


if __name__ == '__main__':
    main()
