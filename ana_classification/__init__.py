""" Main project module """
from .preprocessing import preprocess
from .segmentation import segmentate, SegmentationResult, Segment
from .classification import BaseClassifier, RandomClassifier, ConvNetClassifier, is_negative
from .pipeline import classify_image
