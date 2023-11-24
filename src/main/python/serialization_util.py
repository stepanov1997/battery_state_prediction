import os
import json
import dill as pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score


class SerializationUtil:
    """
    The SerializationUtil class provides static methods for serialization tasks related to machine learning models
    and their associated data. It facilitates the saving of preprocessing pipelines and trained model estimators
    along with their performance statistics. This utility is essential for scenarios requiring model persistence,
    such as deploying models into production or for later analysis and reproducibility of results.

    Methods:
        save_preprocessor: Serializes and saves a preprocessing pipeline to a file.
        save_trained_estimator: Serializes and saves a trained model estimator and its performance statistics.
    """

    @staticmethod
    def save_preprocessor(directory, preprocessor):
        """
        Saves the preprocessing pipeline to a file using pickle for later use
        :param directory: is the path where the preprocessor will be saved
        :param preprocessor: is the pipeline object to be serialized
        """
        with open(os.path.join(directory, 'preprocessor.pkl'), 'wb') as file:
            pickle.dump(preprocessor, file)

    @staticmethod
    def save_trained_estimator(subfolder, name, estimator, statistics):
        """
        Saves the trained estimator (machine learning model) and its statistics.
        :param subfolder: is the directory where the estimator and statistics will be saved
        :param name: is the name of the estimator
        :param estimator: is the model object to be serialized
        :param statistics: contains metrics and information about the model's performance
        """
        # Creating a subfolder for storing the estimators if it doesn't exist
        estimators_subfolder = os.path.join(subfolder, "estimators")

        if not os.path.exists(estimators_subfolder):
            os.makedirs(estimators_subfolder)

        # Filename for the serialized model includes its name and MSE (Mean Squared Error) for easy identification
        estimator_filename = f'estimator_{statistics["mse"]}_{name}'

        # Serializing and saving the estimator using pickle
        with open(os.path.join(estimators_subfolder, f'{estimator_filename}.pkl'), 'wb') as file:
            pickle.dump(estimator, file)

        # Saving the statistics of the estimator in a JSON file for easy readability
        with open(os.path.join(estimators_subfolder, f'{estimator_filename}_statistics.json'), 'w') as file:
            json.dump(statistics, file)
