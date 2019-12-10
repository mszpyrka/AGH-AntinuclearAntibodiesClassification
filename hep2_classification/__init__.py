""" Main project module """
from .preprocessing import preprocess
from .segmentation import segmentate, SegmentationResult, Segment
from .classification import BaseClassifier, RandomClassifier
from .pipeline import classify_image
