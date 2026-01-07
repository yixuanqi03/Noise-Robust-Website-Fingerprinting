import torch
from torch import nn


class DF(nn.Module):
    """
    This class defines the interface of a classical Website Fingerprinting model used as a baseline in the experimental framework.
    """

    def __init__(self, num_classes=100):
        super(DF, self).__init__()
        self.num_classes = num_classes

        # Feature extractor placeholder
        self.feature_extractor = None  # defined in the full version

        # Classifier placeholder
        self.classifier = None  # defined in the full version

    def forward(self, x):
        raise NotImplementedError(
            "Forward logic is intentionally omitted."
        )
