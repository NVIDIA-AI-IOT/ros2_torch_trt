import unittest
from live_classifier.live_classifier_helper import WebcamClassifier


class MockParameter:
    def __init__(self, value):
        self.value = value


class TestCreateClassificationModel(unittest.TestCase):
    def test_invalid_model_name_raises_value_error(self):
        dummy_self = object()
        bad_param = MockParameter("not_a_valid_model")
        with self.assertRaisesRegex(ValueError, "Invalid model selection"):
            WebcamClassifier.create_classification_model(dummy_self, bad_param)


