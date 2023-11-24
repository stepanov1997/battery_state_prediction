from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten


class NeuralNetworkGenerator:
    """
    The NeuralNetworkGeneration class provides static methods for creating different types of neural network models.
    This class serves as a utility for generating Multi-Layer Perceptron (MLP), Long Short-Term Memory (LSTM),
    and Convolutional Neural Network (CNN) models, which can be used for various machine learning tasks.
    """
    @staticmethod
    def generate_mlp_model(input_shape):
        """
        Generates a Multi-Layer Perceptron (MLP) Neural Network model.

        :param input_shape: The shape of the input data.
        :type input_shape: int
        :return: A function that creates an MLP model when called with the specified parameters.
        :rtype: function
        """
        def model_function(optimizer='adam', neurons_layer_1=32, neurons_layer_2=16, activation='relu'):
            model = Sequential()
            model.add(Dense(neurons_layer_1, activation=activation, input_shape=(input_shape,)))
            model.add(Dense(neurons_layer_2, activation=activation))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        return model_function

    @staticmethod
    def generate_lstm_model(input_shape):
        """
        Generates a Long Short-Term Memory (LSTM) Neural Network model.

        :param input_shape: The shape of the input data.
        :type input_shape: int
        :return: A function that creates an LSTM model when called with the specified parameters.
        :rtype: function
        """
        def model_function(lstm_units=20, dense_units=10, activation='relu', optimizer='adam'):
            model = Sequential()

            model.add(LSTM(lstm_units, activation=activation, input_shape=(input_shape, 1)))
            model.add(Dense(dense_units, activation=activation))
            model.add(Dense(1, activation='linear'))

            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        return model_function

    @staticmethod
    def create_cnn_model(input_shape):
        """
        Generates a Convolutional Neural Network (CNN) model.

        :param input_shape: The shape of the input data.
        :type input_shape: int
        :return: A function that creates a CNN model when called with the specified parameters.
        :rtype: function
        """
        def model_function(filters=32, kernel_size=3, dense_units=10, activation='relu', optimizer='adam'):
            model = Sequential()

            model.add(Conv1D(filters, kernel_size=kernel_size, activation=activation, input_shape=(input_shape, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(dense_units, activation=activation))
            model.add(Dense(1, activation='linear'))

            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        return model_function
