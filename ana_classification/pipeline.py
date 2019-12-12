""" Defines function that puts given image thru whole process. """
import numpy as np

from ana_classification import ConvNetClassifier, preprocess, segmentate, is_negative
from ana_classification.preprocessing import _normalize

classifier = ConvNetClassifier()


def classify_image(img: np.ndarray) -> str:
    """
    Returns name of class detected for given image.

    :param img: img
    :return: name of class
    """
    # preprocess image (skipping normalization in order to detect negatives)
    img = preprocess(img, normalize=False)

    # check for negatives
    if is_negative(img):
        return 'NEG'

    # apply normalization manually
    img = _normalize(img)

    # segmentate image into cells
    seg = segmentate(img)

    # classify cells
    results = classifier.classify(list(seg.cells))

    # classify image based on cells classification
    result = classifier.merge_results(results)

    # return name of best class
    return classifier.classes[result.argmax()]
