""" Module containing all classification related functions. """
from abc import ABC, abstractmethod
from typing import List, Iterable

import numpy as np
import cv2
import os

import tensorflow.keras as keras

# ==========================================================
#  CONVOLUTIONAL NEURAL NETWORK CLASSIFIER SETTINGS
# ==========================================================
# size of an image that is fed to the network
CLASSIFICATION_IMG_SIZE = (96, 96)
# file containing saved network model
CLASSIFICATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'resources', 'convnet-model-v1.h5')


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

    def classify(self, images: List[np.ndarray]) -> np.ndarray:
        result = np.random.rand(len(images), len(self.classes))
        result /= result.sum(axis=1).reshape((-1, 1))
        return result

    @property
    def classes(self) -> List[str]:
        return ['ACA', 'AMA', 'DOT', 'FIB']


class ConvNetClassifier(BaseClassifier):

    def __init__(self):
        self._model = keras.models.load_model(CLASSIFICATION_MODEL_FILE)

    @staticmethod
    def _preprocess(img):
        def middle_crop(cell):
            h, w = cell.shape
            shorter = min(h, w)

            h_skip = (h - shorter) // 2
            v_skip = (w - shorter) // 2

            return cell[h_skip:h_skip + shorter, v_skip:v_skip + shorter]

        img = cv2.resize(middle_crop(img), CLASSIFICATION_IMG_SIZE)
        img = img[..., np.newaxis]
        return img / 255

    def classify(self, images: List[np.ndarray]) -> np.ndarray:
        X = np.array([ConvNetClassifier._preprocess(i) for i in images])
        return self._model.predict(X)

    @property
    def classes(self) -> List[str]:
        return ['ZIA', 'HOM', 'ACA', 'FIB']
