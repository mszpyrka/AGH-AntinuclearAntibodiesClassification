""" Defines function that puts given image thru whole process. """
import numpy as np

from hep2_classification import RandomClassifier, preprocess, segmentate

classifier = RandomClassifier()


def classify_image(img: np.ndarray) -> str:
    """
    Returns name of class detected for given image.

    :param img: img
    :return: name of class
    """
    img = preprocess(img)
    seg = segmentate(img)
    results = classifier.classify(list(seg.cells))
    result = classifier.merge_results(results)
    best_class = classifier.classes[result.argmax()]
    return best_class
