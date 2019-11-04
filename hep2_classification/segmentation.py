""" Module containing all segmentation related functions. """
from typing import List, Tuple, NamedTuple

import cv2 as cv
import numpy as np
from skimage import measure, morphology, segmentation as segment
from scipy import ndimage

# ==========================================================
#  SETTINGS
# ==========================================================
AK_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
AK_AVG_MIN = 80.0
AK_AVG_MAX = 100.0
AK_DELTA_MIN = 400.0
AK_DELTA_MAX = 4000.0
THRESHOLD = 240.0
MC_KERNEL = AK_KERNEL
SEEDS_STRUCT = np.ones((3, 3))
SNAKES_SIGMA = 5
SNAKES_ITERATIONS = 5
SNAKES_BALLOON = 1
SNAKES_THRESHOLD = 0.6
SNAKES_SMOOTHING = 0
LMAX_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
LMAX_FRACTION = 0.95
LMAX_THRESHOLD = 10
LMAX_CONNECT_DISTANCE = 10


# ==========================================================
#  PARTIALS
# ==========================================================
def _adaptive_kernel(img: np.ndarray, kernel: np.ndarray = AK_KERNEL,
                     min_avg: float = AK_AVG_MIN, max_avg: float = AK_AVG_MAX,
                     min_delta: float = AK_DELTA_MIN, max_delta: float = AK_DELTA_MAX
                     ) -> np.ndarray:
    """
    Applies given kernel while bin-searching for delta value that results in image
    that average value is in specified range.

    :param img: image to apply kernel to
    :param kernel:
    :param min_avg: minimum value that is accepted while bin-searching
    :param max_avg: maximum value that is accepted while bin-searching
    :param min_delta: minimum value of bin-search range
    :param max_delta: maximum value of bin-search range
    :return: result of applying kernel, if delta that falls in given range was not found throws exception
    """

    while max_delta - min_delta > 50:

        # apply kernel and calculate avg
        avg_delta = (min_delta + max_delta) / 2.0
        kerneled = cv.filter2D(img, -1, kernel, delta=-avg_delta)
        avg = np.average(kerneled)

        # bin search step
        if avg > max_avg:
            min_delta = avg_delta
        elif avg < min_avg:
            max_delta = avg_delta
        else:
            return kerneled

    # if bin search failed
    raise ValueError('Failed to find kernel in given delta range')


def _threshold(img: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """ Applies threshold to given image returning array of ones and zeroes. """
    return (img > threshold).astype(np.uint8)


def _morph_closing(img: np.ndarray, kernel: np.ndarray = MC_KERNEL) -> np.ndarray:
    """ Applies morphological closing to given binary image. """
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)


def _label(img: np.ndarray) -> np.ndarray:
    """ Applies labeling to given binary image returning array in which each group of pixels has the same value. """
    return measure.label(img)


def _distance_transform(img: np.ndarray) -> np.ndarray:
    """ Applies distance transform by calculating distance form each pixel to nearest pixel that evaluates to false. """
    return ndimage.distance_transform_edt(img > 0)


def _local_maximums(img: np.ndarray, kernel: np.ndarray = LMAX_KERNEL, fraction: float = LMAX_FRACTION,
                    global_threshold: float = LMAX_THRESHOLD, connect_distance: int = LMAX_CONNECT_DISTANCE):
    """ Finds local maximums. """
    foreground_mask = img > global_threshold

    # apply maximum filter
    maxed = ndimage.maximum_filter(img, footprint=kernel, mode='reflect')

    # select only interesting peaks
    peaks = (img >= fraction * maxed) * foreground_mask

    # join peaks that are close to each other
    dilation_struct = cv.getStructuringElement(cv.MORPH_ELLIPSE, (connect_distance, connect_distance))
    peaks_dilated = cv.morphologyEx(peaks.astype('uint8'), cv.MORPH_DILATE, dilation_struct)

    # return only peaks that belong are within mask
    return peaks_dilated * foreground_mask


def _watershed_seeds(dists: np.ndarray, struct: np.ndarray = SEEDS_STRUCT) -> np.ndarray:
    """ Finds seeds for watershed algorithm by local max in given distances. """
    local_maxes = _local_maximums(dists)
    labeled_maxes = ndimage.label(local_maxes, structure=struct)[0]
    return labeled_maxes.astype(np.uint8)


def _watershed(labels: np.ndarray) -> np.ndarray:
    """ Applies watershed algorithm on given labels. """
    dists = _distance_transform(labels)
    seeds = _watershed_seeds(dists)
    _, markers = cv.connectedComponents(seeds)
    water = morphology.watershed(-dists, markers)
    return water * (dists > 0)


def _morph_snakes(img: np.ndarray, labels: np.ndarray,
                  sigma: int = SNAKES_SIGMA, iterations: int = SNAKES_ITERATIONS, balloon: int = SNAKES_BALLOON,
                  threshold: float = SNAKES_THRESHOLD, smoothing: float = SNAKES_SMOOTHING
                  ) -> List[np.ndarray]:
    """ Applies morphological active contour method to given image starting from given labels. """
    gradient = ndimage.gaussian_gradient_magnitude(img.astype(np.float32), sigma=sigma)
    return [
        segment.morphological_geodesic_active_contour(
            gradient, iterations, labels == region_id, smoothing=smoothing, balloon=balloon, threshold=threshold
        )
        for region_id in range(1, np.amax(labels))
    ]


def _crop_image(img: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Crops given image by removing rows and columns that all evaluate to false.

    :param img: image to crop
    :return: left-top coordinates of cropped region and region itself
    """
    rows, cols = np.where(img)
    xs = slice(rows.min(), rows.max()+1)
    ys = slice(cols.min(), cols.max()+1)
    return (xs.start, ys.start), img[xs, ys]


# ==========================================================
#  FINAL FUNCTION
# ==========================================================
class SegmentationResult(NamedTuple):
    img: np.ndarray
    offsets: List[Tuple[int, int]]
    masks: List[np.ndarray]

    @property
    def boxes(self):
        for (x, y), mask in zip(self.offsets, self.masks):
            yield (x, y, mask.shape[0], mask.shape[1])

    @property
    def cells(self):
        for x, y, dx, dy in self.boxes:
            yield self.img[x:x+dx, y:y+dy]

    @property
    def cells_masked(self):
        for cell, mask in zip(self.cells, self.masks):
            yield cell * mask

    @property
    def masks_full(self):
        for (x, y, dx, dy), mask in zip(self.boxes, self.masks):
            m = np.zeros_like(self.img)
            m[x:x+dx, y:y+dy] = mask
            yield m


def segmentation(img: np.ndarray) -> SegmentationResult:

    # binary image creation
    img_processed = _adaptive_kernel(img)
    img_processed = _threshold(img_processed)
    img_processed = _morph_closing(img_processed)

    # labels creation
    labels = _label(img_processed)
    labels = _watershed(labels)

    # contours detection
    masks = _morph_snakes(img, labels)
    masks = [_crop_image(mask) for mask in masks]

    return SegmentationResult(img, *zip(*masks))
