""" Module containing all pre-processing related functions. """
from typing import Tuple

import numpy as np
import cv2 as cv


def _reduce_channels(img: np.ndarray) -> np.ndarray:
    """ Reduces channels of given three-channel image to one-channel image. """
    return cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)


def _resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """ Resizes given image to given size using cubic interpolation. """
    return cv.resize(img, dsize=size, interpolation=cv.INTER_CUBIC)


def _denoise(img: np.ndarray, h: float) -> np.ndarray:
    """ Removes noise from given image. Parameter `h` sets filter strength. """
    return cv.fastNlMeansDenoising(img, h=h)


def _normalize(img: np.ndarray, result_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """ Normalizes given image by performing simple interpolation to given range. """
    return np.interp(img, (img.min(), img.max()), result_range).astype(np.uint)
