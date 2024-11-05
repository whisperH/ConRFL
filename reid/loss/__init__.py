from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss, SoftTripletLoss_weight, triplet_loss
from .crossentropy import CrossEntropyLabelSmooth, CrossEntropyLabelSmooth_weighted, cross_entropy_loss
from .nonlap import NonlapLoss
__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'CrossEntropyLabelSmooth_weighted',
    'SoftTripletLoss_weight',
    'NonlapLoss',
    'triplet_loss',
    'cross_entropy_loss',

]
