from data_processor import DataProcessor
from model_trainer import ModelTrainer
from results_processor import ResultsProcessor
from serialization_util import SerializationUtil


class MainController:
    """
    The MainController class serves as the central controller for the application, managing the end-to-end process
    from data preprocessing, through model training, to results processing and evaluation.

    It integrates several components including data processing, model training, and results handling,
    facilitating a streamlined workflow for machine learning model development and evaluation.

    Attributes: data_processor (DataProcessor): An instance of the DataProcessor class for handling data
    preprocessing. serialization_util (SerializationUtil): An instance of the SerializationUtil class for handling
    serialization tasks. model_trainer (ModelTrainer): An instance of the ModelTrainer class for managing the
    training of various machine learning models.
    """

    def __init__(self, data_directory, estimators_data_retreiver):
        """
        Initializes the MainController class.

        :param data_directory: The directory containing the data for model training.
        :type data_directory: str
        :param estimators_data_retriever: Data retriever that containing various machine learning estimators and their
               configurations.
        :type estimators_data_retreiver: function
        """

        self.data_directory = data_directory
        self.results_processor = ResultsProcessor()
        self.data_processor = DataProcessor(data_directory)
        self.serialization_util = SerializationUtil()
        self.model_trainer = ModelTrainer(self.serialization_util, estimators_data_retreiver)

    def run(self):
        """
        Executes the main process flow of the application. This includes setting up result folders,
        preprocessing data, training models, evaluating performance, and storing results.
        """

        # Setting up a directory to store the results of the training process, including a timestamp
        results_directory, timestamp = self.results_processor.setup_result_folders(self.data_directory)

        # Preprocessing the data and splitting it into training and testing sets
        preprocessing_pipeline, X_train, y_train, X_test, y_test = self.data_processor.preprocess_data()

        # Saving the preprocessing pipeline for future use, ensuring consistency in data processing
        self.serialization_util.save_preprocessor(results_directory, preprocessing_pipeline)

        # Main training process: training various models and finding the best performing model
        best_results = self.model_trainer.main_model_training_process(results_directory, X_train, y_train, X_test,
                                                                      y_test)
        (results, best_global_estimator_name, best_global_estimator, best_global_mse, best_global_r2) = best_results

        # Processing and storing the training results, including generating and saving performance charts
        self.results_processor.save_training_results(results_directory, results)
        self.results_processor.generate_and_save_performance_charts(results_directory, results)

        # Renaming the results folder to include the name and performance of the best model
        self.results_processor.rename_folder_to_contain_best_result(results_directory,best_global_estimator_name,
                                                                    best_global_mse)
