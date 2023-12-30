"""
This module implements Trip Advisor and Amazon review dataset managers

Author: Michal Glos (xglosm01)
ZPJa 2023 - FIT VUT
"""
from datasets.dummyDatasets import SummingFloatDataset, SummingIntDataset
from datasets.amznDS import AmazonDataset
from datasets.tripAdvisorDS import TripAdvisorDataset

DATASETS = {
    SummingIntDataset.name: SummingIntDataset,
    SummingFloatDataset.name: SummingFloatDataset,
    AmazonDataset.name: AmazonDataset,
    TripAdvisorDataset.name: TripAdvisorDataset,
}
