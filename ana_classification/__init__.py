""" Main project module """
from .preprocessing import preprocess
from .segmentation import segmentate, SegmentationResult, Segment
from .classification import BaseCellClassifier, RandomCellClassifier, ConvNetCellClassifier, NegClassifier
from .pipeline import classify_image
