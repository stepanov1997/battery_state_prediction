from main_controller import MainController
from grid_params import load_estimators_data

if __name__ == '__main__':
    # Defining the directory where the battery data is stored
    data_directory = 'C:\\Users\\stepa\\PycharmProjects\\battery_state_prediction\\data'

    # Loading the configuration or parameters for various estimators (machine learning models)
    estimators_data_retriever = lambda input_shape: load_estimators_data(input_shape)

    # Creating an instance of the MainController class with the data directory and estimators data
    main_controller = MainController(data_directory, estimators_data_retriever)

    # Running the main process which includes data preprocessing, model training, and result processing
    main_controller.run()
