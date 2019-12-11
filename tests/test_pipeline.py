""" Tests of whole process. """

import glob
import unittest

import cv2 as cv

from ana_classification import classify_image


class TestPipeline(unittest.TestCase):

    def _test_classify(self, paths, expected):
        for path in paths:
            self.assertEqual(expected, classify_image(cv.imread(path)), msg=f'Tested image: {path}')

    def test_classify_HOM(self):
        self._test_classify(glob.glob('resources/HOM-*'), 'HOM')

    def test_classify_NEG(self):
        self._test_classify(glob.glob('resources/NEG-*'), 'NEG')

    def test_classify_FIB(self):
        self._test_classify(glob.glob('resources/FIB-*'), 'FIB')

    def test_classify_ACA(self):
        self._test_classify(glob.glob('resources/ACA-*'), 'ACA')

    def test_classify_ZIA(self):
        self._test_classify(glob.glob('resources/ZIA-*'), 'ZIA')
