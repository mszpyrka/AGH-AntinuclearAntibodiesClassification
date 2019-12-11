""" Tests of classification process. """

import unittest

import numpy as np

from ana_classification import ConvNetClassifier


class TestClassification(unittest.TestCase):

    def setUp(self):
        self.classifier = ConvNetClassifier()

    def test_cnn_classes(self):
        self.assertListEqual(['ZIA', 'HOM', 'ACA', 'FIB'], self.classifier.classes)

    def test_cnn_middle_crop(self):
        img = np.random.randint(0, 255, (100, 10))
        img_cropped = self.classifier._middle_crop(img)
        self.assertTupleEqual((10, 10), img_cropped.shape)
        self.assertTrue(np.all(img[45:55, :] == img_cropped))

        img = np.random.randint(0, 255, (10, 100))
        img_cropped = self.classifier._middle_crop(img)
        self.assertTupleEqual((10, 10), img_cropped.shape)
        self.assertTrue(np.all(img[:, 45:55] == img_cropped))

        img = np.random.randint(0, 255, (100, 100))
        img_cropped = self.classifier._middle_crop(img)
        self.assertTupleEqual(img.shape, img_cropped.shape)
        self.assertTrue(np.all(img == img_cropped))

    def test_cnn_preprocess(self):
        img = np.random.randint(0, 255, (100, 73)).astype('uint8')
        img_processed = self.classifier._preprocess(img)

        self.assertTupleEqual((96, 96, 1), img_processed.shape)
        self.assertTrue(np.all(img_processed >= 0.0))
        self.assertTrue(np.all(img_processed <= 1.0))

    def test_cnn_classify(self):
        images = [np.random.rand(100, 100) for _ in range(3)]
        results = self.classifier.classify(images)

        self.assertTupleEqual((3, 4), results.shape)
        self.assertTrue(np.all(results >= 0))
        self.assertTrue(np.allclose(results.sum(axis=1), np.ones((1, 3))))
