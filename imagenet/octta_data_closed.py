import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from octta_data import (
    _build_eval_transform,
    load_imagenetc_with_ood,
    load_imagenetl_with_ood,
    seed_worker,
)

logger = logging.getLogger(__name__)


_DATASET_ALIASES = {
    "imagenet-c": "imagenet-c",
    "imagenet_c": "imagenet-c",
    "imagenetc": "imagenet-c",
    "imagenet": "imagenet-c",
    "laion-c": "laion-c",
    "laion_c": "laion-c",
    "laionc": "laion-c",
    "laion": "laion-c",
    "imagenet-l": "laion-c",
    "imagenet-sketch": "imagenet-sketch",
    "imagenet_sketch": "imagenet-sketch",
    "sketch": "imagenet-sketch",
    "imagenet-a": "imagenet-a",
    "imagenet_a": "imagenet-a",
    "a": "imagenet-a",
    "imagenet-r": "imagenet-r",
    "imagenet_r": "imagenet-r",
    "r": "imagenet-r",
}


def normalize_closed_set_dataset_name(cfg):
    dataset_name = str(cfg.CORRUPTION.DATASET).strip().lower()

    if dataset_name in {"imagenet", "imagenet-c", "imagenetc", "imagenet_c"}:
        benchmark = str(getattr(cfg.CORRUPTION, "ID_BENCHMARK", "imagenet_c")).strip().lower()
        if benchmark in {"laion_c", "laion-c", "laionc", "laion", "imagenet-l"}:
            return "laion-c"
        return "imagenet-c"

    if dataset_name not in _DATASET_ALIASES:
        raise ValueError(
            f"Unsupported closed-set dataset '{cfg.CORRUPTION.DATASET}'. "
            "Supported values are: imagenet-c / laion-c / imagenet-a / imagenet-sketch / imagenet-r"
        )
    return _DATASET_ALIASES[dataset_name]



def _build_loader(dataset, cfg, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(cfg.RNG_SEED)

    return DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )



def _subset_dataset(dataset, num_ex, rng_seed):
    if num_ex == -1 or len(dataset) <= num_ex:
        return dataset

    indices = np.arange(len(dataset))
    rng = np.random.RandomState(rng_seed)
    rng.shuffle(indices)
    subset_indices = indices[:num_ex]
    return Subset(dataset, subset_indices)



def _project_root_candidates():
    current_dir = Path(__file__).resolve().parent
    return [
        current_dir,
        Path.cwd(),
        current_dir.parent,
    ]



def _resolve_aux_file(*parts):
    relative_path = Path(*parts)
    candidates = []
    for root in _project_root_candidates():
        candidates.append(root / relative_path)
        candidates.append(root / "robustbench" / "data" / relative_path.name)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find auxiliary file '{relative_path}'. Checked: {candidates}"
    )



def _load_imagenet_class_to_idx():
    mapping_path = _resolve_aux_file("robustbench", "data", "imagenet_class_to_id_map.json")
    with open(mapping_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ImageNetSketchDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.loader = default_loader

        rel_list_path = _resolve_aux_file("robustbench", "data", "sketchPath_no_labels_validated.txt")
        class_to_idx = _load_imagenet_class_to_idx()

        self.samples = []
        with open(rel_list_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                rel_path = raw_line.strip()
                if not rel_path:
                    continue

                wnid = rel_path.split("/")[0]
                if wnid not in class_to_idx:
                    continue

                abs_path = self.root / rel_path
                if abs_path.is_file():
                    self.samples.append((str(abs_path), class_to_idx[wnid]))

        if not self.samples:
            raise FileNotFoundError(
                f"Found no validated sketch samples under {self.root} using list {rel_list_path}"
            )

        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target



def _build_original_index_imagefolder(root, transform, expected_num_classes=None):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found at {root}")

    full_class_to_idx = _load_imagenet_class_to_idx()
    valid_class_names = [
        entry.name for entry in os.scandir(root) if entry.is_dir() and entry.name in full_class_to_idx
    ]

    correct_class_to_idx = {
        class_name: full_class_to_idx[class_name]
        for class_name in valid_class_names
    }

    if expected_num_classes is not None and len(correct_class_to_idx) != expected_num_classes:
        logger.warning(
            "Expected %d classes under %s, but found %d mapped classes.",
            expected_num_classes,
            root,
            len(correct_class_to_idx),
        )

    dataset = ImageFolder(root, transform=transform)

    samples = []
    classes = sorted(correct_class_to_idx, key=correct_class_to_idx.get)
    for class_name in classes:
        class_idx = correct_class_to_idx[class_name]
        class_dir = root / class_name
        for file_name in sorted(os.listdir(class_dir)):
            file_path = class_dir / file_name
            if file_path.is_file():
                samples.append((str(file_path), class_idx))

    dataset.class_to_idx = correct_class_to_idx
    dataset.classes = classes
    dataset.samples = samples
    dataset.targets = [target for _, target in samples]
    if hasattr(dataset, "imgs"):
        dataset.imgs = dataset.samples
    return dataset



def load_imagenet_sketch_closed(cfg):
    transforms_test = _build_eval_transform(cfg)
    sketch_root = Path(cfg.DATA_DIR) / "ImageNet-Sketch" / "sketch"

    dataset = ImageNetSketchDataset(sketch_root, transforms_test)
    dataset = _subset_dataset(dataset, cfg.CORRUPTION.NUM_EX, cfg.RNG_SEED)

    logger.info("Loaded %d ImageNet-Sketch samples from %s", len(dataset), sketch_root)
    if cfg.CORRUPTION.NUM_OOD_SAMPLES != 0:
        logger.warning("ImageNet-Sketch closed-set entry ignores NUM_OOD_SAMPLES=%s", cfg.CORRUPTION.NUM_OOD_SAMPLES)

    return _build_loader(dataset, cfg, drop_last=False)



def load_imagenet_a_closed(cfg):
    transforms_test = _build_eval_transform(cfg)
    dataset_root = Path(cfg.DATA_DIR) / "ImageNet-A" / "imagenet-a"

    dataset = _build_original_index_imagefolder(
        dataset_root,
        transforms_test,
        expected_num_classes=200,
    )
    dataset = _subset_dataset(dataset, cfg.CORRUPTION.NUM_EX, cfg.RNG_SEED)

    logger.info("Loaded %d ImageNet-A samples from %s", len(dataset), dataset_root)
    if cfg.CORRUPTION.NUM_OOD_SAMPLES != 0:
        logger.warning("ImageNet-A closed-set entry ignores NUM_OOD_SAMPLES=%s", cfg.CORRUPTION.NUM_OOD_SAMPLES)

    return _build_loader(dataset, cfg, drop_last=False)



def load_imagenet_r_closed(cfg):
    transforms_test = _build_eval_transform(cfg)
    dataset_root = Path(cfg.DATA_DIR) / "ImageNet-R" / "imagenet-r"

    dataset = _build_original_index_imagefolder(
        dataset_root,
        transforms_test,
        expected_num_classes=200,
    )
    dataset = _subset_dataset(dataset, cfg.CORRUPTION.NUM_EX, cfg.RNG_SEED)

    logger.info("Loaded %d ImageNet-R samples from %s", len(dataset), dataset_root)
    if cfg.CORRUPTION.NUM_OOD_SAMPLES != 0:
        logger.warning("ImageNet-R closed-set entry ignores NUM_OOD_SAMPLES=%s", cfg.CORRUPTION.NUM_OOD_SAMPLES)

    return _build_loader(dataset, cfg, drop_last=False)



def load_closed_set_eval_loader(cfg, severity=None, corruption_type=None):
    dataset_name = normalize_closed_set_dataset_name(cfg)

    if dataset_name == "imagenet-c":
        if severity is None or corruption_type is None:
            raise ValueError("ImageNet-C closed-set evaluation requires severity and corruption_type")
        return load_imagenetc_with_ood(cfg, severity, corruption_type)

    if dataset_name == "laion-c":
        if severity is None or corruption_type is None:
            raise ValueError("LAION-C closed-set evaluation requires severity and corruption_type")
        return load_imagenetl_with_ood(cfg, severity, corruption_type)

    if dataset_name == "imagenet-sketch":
        return load_imagenet_sketch_closed(cfg)

    if dataset_name == "imagenet-a":
        return load_imagenet_a_closed(cfg)

    if dataset_name == "imagenet-r":
        return load_imagenet_r_closed(cfg)

    raise ValueError(f"Unsupported closed-set dataset: {dataset_name}")
