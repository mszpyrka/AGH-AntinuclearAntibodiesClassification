""" Module containing all segmentation related functions. """
from typing import List, Tuple, NamedTuple, Generator

import cv2 as cv
import numpy as np
from skimage import measure, morphology, segmentation as segment
from scipy import ndimage

# ==========================================================
#  SETTINGS
# ==========================================================
# kernel used in first convolution
C_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
# average of values in image that the image is shifted to after convolution
TARGET_AVERAGE = 180.0
# threshold value used after convolution to select regions of interests
THRESHOLD = 240.0
# kernel used in morphological closing
MC_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
# kernel used in creating labels from local maxes
SEEDS_STRUCT = np.ones((3, 3))
# settings of active contour algorithm
SNAKES_SIGMA = 5
SNAKES_ITERATIONS = 5
SNAKES_BALLOON = 1
SNAKES_THRESHOLD = 0.6
SNAKES_SMOOTHING = 0
# settings used in calculating local maxes
LMAX_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
LMAX_FRACTION = 0.95
LMAX_THRESHOLD = 10
LMAX_CONNECT_DISTANCE = 10
# mask size below which cells are filtered out as too small
MIN_CELL_SIZE = 900
# mask size over which cells are filtered out as too big
MAX_CELL_SIZE = 16000


# ==========================================================
#  DATA STRUCTURES
# ==========================================================
class Segment(NamedTuple):
    """ Tuple that represents masked part of image. """
    x: int
    y: int
    mask: np.ndarray

    @property
    def slice_x(self) -> slice:
        """ Returns slice along first axis that match this segment. """
        return slice(self.x, self.x + self.mask.shape[0])

    @property
    def slice_y(self) -> slice:
        """ Returns slice along second axis that match this segment. """
        return slice(self.y, self.y + self.mask.shape[1])

    @property
    def slices(self) -> Tuple[slice, slice]:
        """ Returns slices that match this segment. """
        return self.slice_x, self.slice_y

    @staticmethod
    def from_mask(mask: np.ndarray) -> 'Segment':
        """ Creates segment by cropping given mask. """
        rows, cols = np.where(mask)
        xs = slice(rows.min(), rows.max()+1)
        ys = slice(cols.min(), cols.max()+1)
        return Segment(xs.start, ys.start, mask[xs, ys])


class SegmentationResult(NamedTuple):
    """ Tuple that contains segmentation results while also providing helpful getters. """
    img: np.ndarray
    segments: List[Segment]

    @property
    def cells(self) -> Generator[np.ndarray, None, None]:
        """ Yields segments of original image that contain cells. """
        for seg in self.segments:
            yield self.img[seg.slices]

    @property
    def cells_masked(self) -> Generator[np.ndarray, None, None]:
        """ Yields segments of original image that contain cells with parts that do not belong to cell set to 0. """
        for seg in self.segments:
            yield self.img[seg.slices] * seg.mask

    @property
    def masks_full(self) -> Generator[np.ndarray, None, None]:
        """ Yields cell masks that have shape of original image. """
        for seg in self.segments:
            m = np.zeros_like(self.img)
            m[seg.slices] = seg.mask
            yield m

    def save(self, path: str, compressed: bool = True):
        """ Saves this segmentation result to numpy .npz file. """
        method = np.savez_compressed if compressed else np.savez
        data = {f'mask_{i}': seg.mask for i, seg in enumerate(self.segments)}
        data['img'] = self.img
        data['offsets'] = np.array([(seg.x, seg.y) for seg in self.segments])

        method(path, **data)

    @staticmethod
    def load(path: str) -> 'SegmentationResult':
        """ Loads segmentation result form numpy .npz file. """
        data = np.load(path)
        img = data['img']
        offsets = data['offsets']
        masks = [data[f'mask_{i}'] for i in range(offsets.shape[0])]

        return SegmentationResult(img, [Segment(x, y, mask) for (x, y), mask in zip(offsets, masks)])


# ==========================================================
#  PARTIALS
# ==========================================================
def _convolve(img: np.ndarray, kernel: np.ndarray = C_KERNEL) -> np.ndarray:
    """ Applies convolution using given kernel. """
    return ndimage.convolve(img.astype('float'), kernel, mode='constant')


def _shift_to_average(img: np.ndarray, avg: float = TARGET_AVERAGE) -> np.ndarray:
    """ Shifts values in image so that its average is equal to the given one. """
    return img - np.average(img) + avg


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


def _convex_hull(seg: Segment) -> Segment:
    """ Returns convex hull of given segment. """
    return Segment(seg.x, seg.y, morphology.convex_hull_image(seg.mask))


def _is_touching_edge(img: np.ndarray, seg: Segment) -> bool:
    """ Returns whether given segment touches edges of given image """
    return seg.slice_x.start <= 0 \
        or seg.slice_x.stop >= img.shape[0] \
        or seg.slice_y.start <= 0 \
        or seg.slice_y.stop >= img.shape[1]


def _has_expected_size(seg: Segment, min_size: int = MIN_CELL_SIZE, max_size: int = MAX_CELL_SIZE) -> bool:
    """ Returns whether mask of given segment has size between given values. """
    return min_size <= np.sum(seg.mask) <= max_size


# ==========================================================
#  FINAL FUNCTION
# ==========================================================
def segmentate(img: np.ndarray) -> SegmentationResult:
    """ Performs whole segmentation process of given image. """

    # binary image creation
    img_processed = _convolve(img)
    img_processed = _shift_to_average(img_processed)
    img_processed = _threshold(img_processed)
    img_processed = _morph_closing(img_processed)

    # labels creation
    labels = _label(img_processed)
    labels = _watershed(labels)

    # contours detection
    masks = _morph_snakes(img, labels)

    # segments creation
    segments = map(Segment.from_mask, masks)

    # convex hull
    segments = map(_convex_hull, segments)

    # segments filtering
    segments = filter(lambda s: not _is_touching_edge(img, s), segments)
    segments = filter(lambda s: _has_expected_size(s), segments)

    return SegmentationResult(img, list(segments))
