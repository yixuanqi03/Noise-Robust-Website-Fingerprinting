import torch
from torch import nn


class DF(nn.Module):
    """
    Reference backbone placeholder (DF).

    This class defines the interface of a classical Website Fingerprinting
    model used as a baseline in the experimental framework. Architectural
    details are omitted in the review version.
    """

    def __init__(self, num_classes=100):
        super(DF, self).__init__()
        self.num_classes = num_classes

        # Feature extractor placeholder
        self.feature_extractor = None  # defined in the full version

        # Classifier placeholder
        self.classifier = None  # defined in the full version

    def forward(self, x):
        """
        Forward pass placeholder.
        """
        raise NotImplementedError(
            "Forward logic is omitted in the review version."
        )
