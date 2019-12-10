""" Module containing all classification related functions. """
from abc import ABC, abstractmethod
from typing import List, Iterable

import numpy as np


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
