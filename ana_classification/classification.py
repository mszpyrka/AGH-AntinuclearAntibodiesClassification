""" Module containing all classification related functions. """
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import cv2
import os

# disable info printed by tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

# ==========================================================
#  CONVOLUTION NEURAL NETWORK CLASSIFIER SETTINGS
# ==========================================================
# size of an image that is fed to the network
CLASSIFICATION_IMG_SIZE = (96, 96)
# file containing saved network model
CLASSIFICATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'resources', 'convnet-model-v1.h5')


# ==========================================================
#  CLASSIFIERS
# ==========================================================
class BaseClassifier(ABC):
    """ Interface for other classifiers. """

    @abstractmethod
    def classify(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Returns membership percentages of given images to each of the classes.

        :param images: list of one channel images
        :return: membership percentages if form of numpy array of shape (len(images), len(classes))
        """
        pass

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """
        Returns classes recognized by this classifier.

        :return: Classes in the same order as returned from classify()
        """
        pass

    @staticmethod
    def merge_results(results: np.ndarray) -> np.ndarray:
        """
        Merges results from classify() function by calculating average for each class.

        :param results: membership percentages from classify()
        :return: membership percentages if form of numpy vector of length len(classes)
        """
        return results.sum(axis=0) / results.shape[0]


class RandomClassifier(BaseClassifier):
    """ Random classifier for development purposes. """

    def classify(self, images: List[np.ndarray]) -> np.ndarray:
        """ Returns random membership percentages. """
        result = np.random.rand(len(images), len(self.classes))
        result /= result.sum(axis=1).reshape((-1, 1))
        return result

    @property
    def classes(self) -> List[str]:
        return ['ACA', 'AMA', 'DOT', 'FIB']


class ConvNetClassifier(BaseClassifier):
    """ Convolution neural network based classifier. """

    def __init__(self):
        """ Loads keras model. """
        self._model = keras.models.load_model(CLASSIFICATION_MODEL_FILE)

    def classify(self, images: List[np.ndarray]) -> np.ndarray:
        """ Performs classification using CNN. """
        x = np.array([ConvNetClassifier._preprocess(img) for img in images])
        return self._model.predict(x)

    @property
    def classes(self) -> List[str]:
        return ['ZIA', 'HOM', 'ACA', 'FIB']

    @staticmethod
    def _preprocess(img):
        """ Prepares given cell so that it matches format expected by model. """
        img = ConvNetClassifier._middle_crop(img)
        img = cv2.resize(img, CLASSIFICATION_IMG_SIZE)
        img = img[..., np.newaxis]
        return img / 255

    @staticmethod
    def _middle_crop(cell: np.ndarray) -> np.ndarray:
        """ Crops image from the center of cell. """
        h, w = cell.shape
        shorter = min(h, w)
        h_skip = (h - shorter) // 2
        v_skip = (w - shorter) // 2
        return cell[h_skip:h_skip + shorter, v_skip:v_skip + shorter]
