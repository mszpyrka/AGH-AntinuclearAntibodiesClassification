""" Module containing all pre-processing related functions. """
import os
from typing import Tuple, Optional

import numpy as np
import cv2 as cv

# constant default settings for pre-processing
IMG_SIZE = (800, 600)
TAG_TEMPLATE = cv.imread(os.path.join(os.path.dirname(__file__), 'tag-template-20.tif'))
TAG_SEARCH_RANGE = (slice(70), slice(340))
TAG_MIN_MATCH = 3 * 10**6
TAG_CUTOFF = (slice(55, None), slice(None, None))
DENOISE_STRENGTH = 2


def _match_tag(img: np.ndarray,
               search_range: Tuple[slice, slice] = TAG_SEARCH_RANGE,
               tag_template: np.ndarray = TAG_TEMPLATE
               ) -> np.ndarray:
    """
    Calculates match probability for tag in given image.

    :param img: image to search in
    :param search_range: selector used to restrict search region
    :param tag_template: template to match
    :return: array of probabilities returned by `cv2.matchTemplate`
    """
    return cv.matchTemplate(img[search_range], tag_template, cv.TM_CCOEFF)


def _remove_tag(img: np.ndarray, min_match: float = TAG_MIN_MATCH) -> np.ndarray:
    """
    Removes tag (if it is present) by cutting of top part of the image.

    :param img: image to remove tag from
    :param min_match: minimum match value for which tag will be removed
    :return: image with removed (or not) part containing tag
    """
    if _match_tag(img).max() > min_match:
        return img[TAG_CUTOFF]
    return img


def _reduce_channels(img: np.ndarray) -> np.ndarray:
    """ Reduces channels of given three-channel image to one-channel image. """
    return cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)


def _resize(img: np.ndarray, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """ Resizes given image to given size using cubic interpolation. """
    return cv.resize(img, dsize=size, interpolation=cv.INTER_CUBIC)


def _denoise(img: np.ndarray, h: float = DENOISE_STRENGTH) -> np.ndarray:
    """ Removes noise from given image. Parameter `h` sets filter strength. """
    return cv.fastNlMeansDenoising(img, h=h)


def _normalize(img: np.ndarray, result_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """ Normalizes given image by performing simple interpolation to given range. """
    return np.interp(img, (img.min(), img.max()), result_range).astype(np.uint)


def _equalize_histogram(img: np.ndarray) -> np.ndarray:
    """ Equalizes histogram of image therefor increasing contrast. """
    return cv.equalizeHist(img)
