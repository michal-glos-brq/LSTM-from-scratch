from datasets.dummyDatasets import SummingFloatDataset, SummingIntDataset
from datasets.amznDS import AmazonDataset
from datasets.tripAdvisorDS import TripAdvisorDataset

DATASETS = {
    SummingIntDataset.name: SummingIntDataset,
    SummingFloatDataset.name: SummingFloatDataset,
    AmazonDataset.name: AmazonDataset,
    TripAdvisorDataset.name: TripAdvisorDataset,
}
