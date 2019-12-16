""" Functions for generating overlays with info. """
from typing import Tuple, List

import cv2 as cv
import numpy as np


def draw_overlay(img: np.ndarray, boxes: List[Tuple[Tuple[int, int], Tuple[int, int], str]],
                 results: List[Tuple[float, int, str]]) -> np.ndarray:
    """ Adds an overlay to given image. """

    # create three channel image
    if len(img.shape) < 3:
        img = to_3_channels(img)

    # draw boxes
    for box in boxes:
        img = draw_box(img, *box)

    # add bottom info
    img = draw_image_info(img, results)

    return img


def to_3_channels(img: np.ndarray) -> np.ndarray:
    """ Returns given image but with three channels. """
    return cv.merge([img] * 3)


def draw_box(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
             text: str = "", color=(0, 0, 255), color_text=(0, 0, 0)) -> np.ndarray:
    """ Draws box on given image with given annotation. """
    img = cv.rectangle(img, p1, p2, color)
    img = cv.rectangle(img, p1, (p1[0] + 52, p1[1] + 9), color, cv.FILLED)
    img = cv.putText(img, text, (p1[0], p1[1] + 8), cv.FONT_HERSHEY_SIMPLEX, 0.3, color_text)
    return img


def draw_image_info(img: np.ndarray, results: List[Tuple[float, int, str]], color=(255, 255, 255)) -> np.ndarray:
    """ Adds margin with information to the bottom of image. """
    y = img.shape[0]
    # add margin
    img = cv.copyMakeBorder(img, 0, 50, 0, 0, cv.BORDER_CONSTANT, value=0)
    # display image class in bottom-left
    img = cv.putText(img, results[0][2], (20, y + 35), cv.FONT_HERSHEY_SIMPLEX, 1, color)
    # display counts of classes
    text = ', '.join([f'{name}: {count}' for _, count, name in results])
    img = cv.putText(img, text, (110, y+17), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)
    # display probabilities of classes
    text = ', '.join([f'{name}: {prob:.2f}' for prob, _, name in results])
    img = cv.putText(img, text, (110, y+40), cv.FONT_HERSHEY_SIMPLEX, 0.4, color)
    return img
