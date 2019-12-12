""" Tests of pre-processing """

import unittest

import cv2 as cv

from ana_classification.preprocessing import preprocess, _remove_tag, _reduce_channels, _resize, _denoise, _normalize, \
    _equalize_histogram


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.img = cv.imread('resources/HOM-1.tif')

    def test_remove_tag(self):
        img_removed = _remove_tag(self.img)

        self.assertTupleEqual((self.img.shape[0] - 55, self.img.shape[1], self.img.shape[2]), img_removed.shape)

    def test_reduce_channels(self):
        img_reduced = _reduce_channels(self.img)

        self.assertTupleEqual(self.img.shape[:2], img_reduced.shape)

    def test_resize(self):
        img_resized = _resize(_reduce_channels(self.img))

        self.assertTupleEqual((600, 800), img_resized.shape)

    def test_denoise(self):
        img_denoised = _denoise(_reduce_channels(self.img))

        self.assertTupleEqual(self.img.shape[:2], img_denoised.shape)

    def test_normalize(self):
        img_normalized = _normalize(_reduce_channels(self.img))

        self.assertTupleEqual(self.img.shape[:2], img_normalized.shape)

    def test_equalize_histogram(self):
        img_equalized = _equalize_histogram(_reduce_channels(self.img))

        self.assertTupleEqual(self.img.shape[:2], img_equalized.shape)

    def test_preprocessing(self):
        img_preprocessed = preprocess(self.img)

        self.assertTupleEqual((600, 800), img_preprocessed.shape)
