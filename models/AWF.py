import torch
from torch import nn


class AWF(nn.Module):
    """
    This class defines the interface of a standard Website Fingerprinting backbone used in the experimental framework. 
    """

    def __init__(self, num_classes=100):
        super(AWF, self).__init__()
        self.num_classes = num_classes

        # Feature extractor placeholder
        self.feature_extractor = None  # defined in the full version

        # Classifier placeholder
        self.classifier = None  # defined in the full version

    def forward(self, x):
        """
        In the full implementation, this method performs feature extraction followed by classification.
        """
        raise NotImplementedError(
            "Forward logic is intentionally omitted."
        )
