from torch.utils.data import DataLoader
from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.segmentation_dataset import SegmentationDataset


def get_classification_dataloader(data_dir, batch_size, split, transform):
    dataset = ClassificationDataset(
        data_dir=data_dir,
        split=split,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True
    )
    return loader


def get_segmentation_dataloader(data_dir, batch_size, split, transform):
    dataset = SegmentationDataset(
        data_dir=data_dir,
        split=split,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True
    )
    return loader