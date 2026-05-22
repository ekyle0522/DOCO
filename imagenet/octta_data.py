import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from robustbench.data import CustomImageFolder, PREPROCESSINGS

logger = logging.getLogger(__name__)


class NCropsTransform:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x):
        return [transform(x) for transform in self.transform_list]


def get_stamp_transforms():
    return transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

def _build_eval_transform(cfg):
    adaptation = str(getattr(cfg.MODEL, "ADAPTATION", "")).lower()

    # ImageNet-C / ImageNet-LAION-5K / OOD-C / OOD-L are already 224x224.
    # Avoid the extra 224 -> 256 -> 224 resize/crop.
    transforms_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    if adaptation != "stamp":
        return transforms_test

    num_aug = int(getattr(cfg.STAMP, "NUM_AUG", 15))
    transform_list = [transforms_test]
    transform_aug = transforms.Compose([
        transforms_test,
        get_stamp_transforms(),
    ])
    transform_list.extend([transform_aug for _ in range(num_aug)])

    logger.info(
        "Using STAMP multi-view transform without Res256Crop224: "
        "1 original view + %d augmented views.",
        num_aug,
    )
    return NCropsTransform(transform_list)


IMAGENETC_OOD_DATASET_CONFIG = {
    # Type1: Flat
    "places365":   {"type": "flat",   "path_component": "PLACES365-C",   "sub_path": "val_256"},
    "inaturalist": {"type": "flat",   "path_component": "iNaturalist-C", "sub_path": "images"},
    "sun":         {"type": "flat",   "path_component": "SUN-C",         "sub_path": "images"},
  

    # Type2: Sub_folds
    "textures":          {"type": "folder", "path_component": "Textures-C",                "sub_path": "images"},
    "ninco_ood_classes": {"type": "folder", "path_component": "NINCO_OOD_classes-C",       "sub_path": "images"},
    "ssb-hard":          {"type": "folder", "path_component": "SSB-Hard-C",                "sub_path": "images"},
}

LAION_OOD_DATASET_CONFIG = {
    # Type1: Flat outputs are written directly under corruption/intensity_level_*.
    "places365":   {"type": "flat",   "path_component": "Places365-L-6k",                "sub_path": ""},
    "inaturalist": {"type": "flat",   "path_component": "iNaturalist-L-6k",              "sub_path": ""},
    "sun":         {"type": "flat",   "path_component": "SUN-L-6k",                      "sub_path": ""},

    # Type2: Nested outputs keep class folders directly under corruption/intensity_level_*.
    "textures":          {"type": "folder", "path_component": "Textures-L-6k",                "sub_path": ""},
    "ninco_ood_classes": {"type": "folder", "path_component": "NINCO_OOD_classes-L-6k",       "sub_path": ""},
    "ssb-hard":          {"type": "folder", "path_component": "SSB-Hard-L-6k",                "sub_path": ""},
}


def _normalize_id_dataset_name(name):
    if name is None:
        return "imagenet_c"

    name = str(name).strip().lower()

    imagenetc_alias = {"imagenet_c", "imagenetc", "imagenet-c", "c"}
    laion_alias = {"laion_c", "laionc", "laion-c", "imagenet_laion", "imagenet-l", "laion"}

    if name in imagenetc_alias:
        return "imagenet_c"
    if name in laion_alias:
        return "laion_c"

    raise ValueError(
        f"Unsupported CORRUPTION.ID_BENCHMARK = '{name}'. "
        f"Supported values are: imagenet_c / laion_c"
    )


def _resolve_eval_layout(cfg, severity, corruption_type, forced_id_dataset=None):
    id_dataset_name = forced_id_dataset
    if id_dataset_name is None:
        id_dataset_name = getattr(cfg.CORRUPTION, "ID_BENCHMARK", "imagenet_c")

    id_dataset_name = _normalize_id_dataset_name(id_dataset_name)

    if id_dataset_name == "imagenet_c":
        id_data_folder_path = Path(cfg.DATA_DIR) / "ImageNet-C" / corruption_type / str(severity)
        ood_dataset_config = IMAGENETC_OOD_DATASET_CONFIG
        severity_component = str(severity)
    else:
        id_data_folder_path = (
            Path(cfg.DATA_DIR)
            / "ImageNet-LAION-5K"
            / corruption_type
            / f"intensity_level_{severity}"
        )
        ood_dataset_config = LAION_OOD_DATASET_CONFIG
        severity_component = f"intensity_level_{severity}"

    return id_dataset_name, id_data_folder_path, ood_dataset_config, severity_component


def _build_ood_dataset(ood_data_folder_path, config, transforms_test):
    if config["type"] == "flat":
        return OODImageFolder(ood_data_folder_path, transforms_test)
    elif config["type"] == "folder":
        return RecursiveImageFolderOOD(ood_data_folder_path, transforms_test)
    else:
        raise ValueError(f"Unknown OOD dataset type: {config['type']}")


def _subset_ood_dataset(ood_dataset, cfg):
    num_total_ood = len(ood_dataset)
    indices = np.arange(num_total_ood)

    rng = np.random.RandomState(cfg.RNG_SEED)
    rng.shuffle(indices)

    num_samples_to_take = cfg.CORRUPTION.NUM_OOD_SAMPLES

    if num_samples_to_take != -1 and num_total_ood > num_samples_to_take:
        subset_indices = indices[:num_samples_to_take]
        ood_dataset = Subset(ood_dataset, subset_indices)
    else:
        ood_dataset = Subset(ood_dataset, indices)

    return ood_dataset


def _load_with_ood(cfg, severity, corruption_type, forced_id_dataset=None):
    transforms_test = _build_eval_transform(cfg)

    (
        id_dataset_name,
        id_data_folder_path,
        ood_dataset_config,
        severity_component,
    ) = _resolve_eval_layout(cfg, severity, corruption_type, forced_id_dataset)

    # 1) ID
    id_dataset = CustomImageFolder(id_data_folder_path, transforms_test)
    id_dataset = Subset(id_dataset, np.arange(cfg.CORRUPTION.NUM_EX))
    logger.info(
        "[%s] Loaded %d ID samples from %s",
        id_dataset_name,
        len(id_dataset),
        id_data_folder_path,
    )

    # 2) Closed-set Benchmark
    num_samples_to_take = cfg.CORRUPTION.NUM_OOD_SAMPLES
    if num_samples_to_take == 0:
        logger.info("[%s] num_samples_to_take == 0, Start closed-set evaluation", id_dataset_name)
        return _build_loader(id_dataset, cfg)

    # 3) OOD
    ood_dataset_name = cfg.CORRUPTION.OOD_DATASET.lower()
    if ood_dataset_name not in ood_dataset_config:
        raise ValueError(
            f"[{id_dataset_name}] OOD dataset '{cfg.CORRUPTION.OOD_DATASET}' is not supported. "
            f"Please add it to the corresponding OOD_DATASET_CONFIG."
        )

    config = ood_dataset_config[ood_dataset_name]
    ood_data_folder_path = (
        Path(cfg.DATA_DIR)
        / config["path_component"]
        / corruption_type
        / severity_component
    )
    if config["sub_path"]:
        ood_data_folder_path = ood_data_folder_path / config["sub_path"]

    ood_dataset = _build_ood_dataset(ood_data_folder_path, config, transforms_test)
    logger.info(
        "[%s] Total OOD samples found: %d at %s",
        id_dataset_name,
        len(ood_dataset),
        ood_data_folder_path,
    )

    ood_dataset = _subset_ood_dataset(ood_dataset, cfg)
    logger.info(
        "[%s] Loaded %d OOD samples from %s using '%s' (%s) strategy.",
        id_dataset_name,
        len(ood_dataset),
        ood_data_folder_path,
        ood_dataset_name,
        config["type"],
    )

    combined_dataset = ConcatDataset([id_dataset, ood_dataset])
    return _build_loader(combined_dataset, cfg)


def load_imagenetc_with_ood(cfg, severity, corruption_type):
    return _load_with_ood(cfg, severity, corruption_type, forced_id_dataset="imagenet_c")


def load_imagenetl_with_ood(cfg, severity, corruption_type):
    return _load_with_ood(cfg, severity, corruption_type, forced_id_dataset="laion_c")


def load_eval_with_ood(cfg, severity, corruption_type):
    return _load_with_ood(cfg, severity, corruption_type)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class OODImageFolder(Dataset):
    """
    Load an OOD image directory without class subfolders and assign all labels to -1.
    """
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.samples = [
            p
            for p in Path(self.root).glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]
        if not self.samples:
            raise FileNotFoundError(f"Found no images in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, -1




class RecursiveImageFolderOOD(Dataset):
    """
    Recursively scan all image files under root, ignore empty class directories,
    and assign all labels to -1.
    Suitable for OOD datasets with class subdirectories without relying on
    ImageFolder's empty-class checks.
    """
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.loader = default_loader

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.samples = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMG_EXTENSIONS
        )

        if not self.samples:
            raise FileNotFoundError(f"Found no valid image files under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, -1


def _build_loader(dataset, cfg):
    g = torch.Generator()
    g.manual_seed(cfg.RNG_SEED)

    return DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
