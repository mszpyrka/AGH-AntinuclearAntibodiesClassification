""" Tests of segmentation process. """

import unittest

import numpy as np
import cv2 as cv

from ana_classification import preprocess
from ana_classification.segmentation import _convolve, _shift_to_average, _threshold, _morph_closing, _label, \
    _is_touching_edge, Segment, _has_expected_size, segmentate, SegmentationResult, _convex_hull


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.img = preprocess(cv.imread('resources/HOM-1.tif'))

    def test_convolution(self):
        img_convolved = _convolve(self.img)

        self.assertTupleEqual((600, 800), img_convolved.shape)

    def test_shift_to_average(self):
        img_shifted = _shift_to_average(_convolve(self.img))

        self.assertTupleEqual((600, 800), img_shifted.shape)
        self.assertAlmostEqual(180.0, np.average(img_shifted))

    def test_threshold(self):
        img_threshold = _threshold(_shift_to_average(_convolve(self.img)))

        self.assertTupleEqual((600, 800), img_threshold.shape)
        self.assertListEqual([0, 1], list(np.unique(img_threshold)))

    def test_morph_closing(self):
        img_closed = _morph_closing(_threshold(_shift_to_average(_convolve(self.img))))

        self.assertTupleEqual((600, 800), img_closed.shape)
        self.assertListEqual([0, 1], list(np.unique(img_closed)))

    def test_label(self):
        labels = _label(_morph_closing(_threshold(_shift_to_average(_convolve(self.img)))))

        self.assertTupleEqual((600, 800), labels.shape)
        self.assertEqual(0, np.min(labels))

    def test_from_mask(self):
        mask = np.zeros((100, 100))
        mask[20, 20] = 1

        seg = Segment.from_mask(mask)
        self.assertEqual(20, seg.x)
        self.assertEqual(20, seg.y)
        self.assertTupleEqual((1, 1), seg.mask.shape)

        mask[30, 60] = 1

        seg = Segment.from_mask(mask)
        self.assertEqual(20, seg.x)
        self.assertEqual(20, seg.y)
        self.assertTupleEqual((11, 41), seg.mask.shape)

        mask[0, 5] = 1

        seg = Segment.from_mask(mask)
        self.assertEqual(0, seg.x)
        self.assertEqual(5, seg.y)
        self.assertTupleEqual((31, 56), seg.mask.shape)

    def test_convex_hull(self):
        seg = Segment(13, 41, np.zeros((100, 100)))

        seg_convex = _convex_hull(seg)
        self.assertEqual(seg.x, seg_convex.x)
        self.assertEqual(seg.y, seg_convex.y)
        self.assertTrue(np.all(seg_convex.mask == 0))

        seg.mask[0, 0] = 1
        seg.mask[0, 99] = 1
        seg.mask[99, 0] = 1
        seg.mask[99, 99] = 1

        seg_convex = _convex_hull(seg)
        self.assertEqual(seg.x, seg_convex.x)
        self.assertEqual(seg.y, seg_convex.y)
        self.assertTrue(np.all(seg_convex.mask == 1))

    def test_is_touching_edge(self):
        self.assertTrue(_is_touching_edge(self.img, Segment(0, 0, np.empty((10, 10)))))
        self.assertTrue(_is_touching_edge(self.img, Segment(100, 0, np.empty((10, 10)))))
        self.assertTrue(_is_touching_edge(self.img, Segment(0, 100, np.empty((10, 10)))))
        self.assertTrue(_is_touching_edge(self.img, Segment(590, 790, np.empty((10, 10)))))
        self.assertTrue(_is_touching_edge(self.img, Segment(590, 100, np.empty((10, 10)))))
        self.assertTrue(_is_touching_edge(self.img, Segment(100, 790, np.empty((10, 10)))))
        self.assertFalse(_is_touching_edge(self.img, Segment(100, 100, np.empty((10, 10)))))

        self.assertFalse(_is_touching_edge(self.img, Segment(1, 1, np.empty((10, 10)))))
        self.assertFalse(_is_touching_edge(self.img, Segment(589, 789, np.empty((10, 10)))))

    def test_has_expected_size(self):
        seg = Segment(0, 0, np.zeros((200, 200)))
        self.assertFalse(_has_expected_size(seg))

        seg.mask[10:110, 40:150] = 1
        self.assertTrue(_has_expected_size(seg))

        seg.mask[:, :81] = 1
        self.assertFalse(_has_expected_size(seg))

    def test_segmentate(self):
        result = segmentate(self.img)

        self.assertIsInstance(result, SegmentationResult)
        self.assertTrue(np.all(self.img == result.img))
        self.assertEqual(60, len(result.segments))

        for seg in result.segments:
            self.assertIsInstance(seg, Segment)
            self.assertListEqual([0, 1], list(np.unique(seg.mask)))
            self.assertLessEqual(0, seg.x)
            self.assertLessEqual(0, seg.y)
            self.assertGreater(self.img.shape[0], seg.x + seg.mask.shape[0])
            self.assertGreater(self.img.shape[1], seg.y + seg.mask.shape[1])
