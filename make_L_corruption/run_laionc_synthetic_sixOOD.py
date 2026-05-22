import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/d/stamp_lib/datasets"))

APPLY_SCRIPT_PATH = str(SCRIPT_DIR / "distortions" / "modified_apply_distortions.py")
TILE_DIR_PATH = str(DATA_ROOT / "imagenet_val_tiles.tar")

CORRUPTION_TYPES = [
    "mosaic",
    "sticker",
    "glitched",
    "vertical_lines",
    "geometric_shapes",
    "luminance",
]

INTENSITY_LEVELS = [1, 3]
MIN_VALID_FILE_SIZE_BYTES = 1
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

DATASETS = {
    "iNaturalist": {
        "TASK_NAME": "iNaturalist-L-6k (OOD-Sequential)",
        "SOURCE_ROOT": str(DATA_ROOT / "iNaturalist" / "images"),
        "OUTPUT_ROOT": str(DATA_ROOT / "iNaturalist-L-6k"),
        "FILE_STRUCTURE": "flat",
        "SAMPLING_MODE": "sequential",
        "NUM_SAMPLES": 6000,
    },
    "places365": {
        "TASK_NAME": "Places365-L-Subset (OOD-Sequential)",
        "SOURCE_ROOT": str(DATA_ROOT / "PLACES365" / "val_256"),
        "OUTPUT_ROOT": str(DATA_ROOT / "Places365-L-6k"),
        "FILE_STRUCTURE": "flat",
        "SAMPLING_MODE": "sequential",
        "NUM_SAMPLES": 6000,
    },
    "SUN": {
        "TASK_NAME": "SUN-L-6k (OOD-Sequential)",
        "SOURCE_ROOT": str(DATA_ROOT / "SUN" / "images"),
        "OUTPUT_ROOT": str(DATA_ROOT / "SUN-L-6k"),
        "FILE_STRUCTURE": "flat",
        "SAMPLING_MODE": "sequential",
        "NUM_SAMPLES": 6000,
    },
    "textures": {
        "TASK_NAME": "Textures-L-Subset (OOD-Random)",
        "SOURCE_ROOT": str(DATA_ROOT / "Textures" / "images"),
        "OUTPUT_ROOT": str(DATA_ROOT / "Textures-L-6k"),
        "FILE_STRUCTURE": "nested",
        "SAMPLING_MODE": "random",
        "NUM_SAMPLES": 6000,
        "RNG_SEED": 22,
    },
    "NINCO": {
        "TASK_NAME": "NINCO_OOD_classes (OOD-Random)",
        "SOURCE_ROOT": str(DATA_ROOT / "NINCO" / "NINCO_OOD_classes"),
        "OUTPUT_ROOT": str(DATA_ROOT / "NINCO_OOD_classes-L-6k"),
        "FILE_STRUCTURE": "nested",
        "SAMPLING_MODE": "random",
        "NUM_SAMPLES": 6000,
        "RNG_SEED": 22,
    },
    "SSBHard": {
        "TASK_NAME": "SSB-Hard-L-6k (OOD-Random)",
        "SOURCE_ROOT": str(DATA_ROOT / "SSB-Hard"),
        "OUTPUT_ROOT": str(DATA_ROOT / "SSB-Hard-L-6k"),
        "FILE_STRUCTURE": "nested",
        "SAMPLING_MODE": "random",
        "NUM_SAMPLES": 6000,
        "RNG_SEED": 22,
    },
}

# Comment out datasets here to run only a subset.
ENABLED_DATASETS = [
    "iNaturalist",
    "places365",
    "SUN",
    "textures",
    "NINCO",
    "SSBHard",
]


def is_supported_source_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def list_source_images_in_dir(dir_path: Path):
    return sorted([p for p in Path(dir_path).iterdir() if is_supported_source_image(p)])


def list_existing_outputs_in_dir(dir_path: Path):
    return sorted([p for p in Path(dir_path).iterdir() if p.is_file()])


def select_files_for_corruption(cfg):
    print(f"Selecting {cfg['NUM_SAMPLES']} files using '{cfg['SAMPLING_MODE']}' mode...")

    source_root = Path(cfg["SOURCE_ROOT"])
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    all_files = []
    if cfg["FILE_STRUCTURE"] == "nested":
        class_folders = sorted([d for d in source_root.iterdir() if d.is_dir()])
        for class_folder in class_folders:
            all_files.extend(list_source_images_in_dir(class_folder))
    elif cfg["FILE_STRUCTURE"] == "flat":
        all_files = list_source_images_in_dir(source_root)
    else:
        raise ValueError(f"Unknown FILE_STRUCTURE: {cfg['FILE_STRUCTURE']}")

    print(f"Found {len(all_files)} total supported images in source directory.")

    if len(all_files) < cfg["NUM_SAMPLES"]:
        print(
            f"[Warning] Requested {cfg['NUM_SAMPLES']} samples, but only "
            f"{len(all_files)} supported images are available. Using all available files."
        )
        return all_files

    if cfg["SAMPLING_MODE"] == "sequential":
        selected_files = all_files[: cfg["NUM_SAMPLES"]]
        print(f"Selected the first {len(selected_files)} files sequentially.")
        return selected_files

    if cfg["SAMPLING_MODE"] == "random":
        rng_seed = cfg["RNG_SEED"]
        print(f"Using random seed: {rng_seed}")
        rng = np.random.RandomState(rng_seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        selected_indices = indices[: cfg["NUM_SAMPLES"]]
        selected_files = [all_files[i] for i in selected_indices]
        print(f"Selected {len(selected_files)} files randomly.")
        return selected_files

    raise ValueError(f"Unknown SAMPLING_MODE: {cfg['SAMPLING_MODE']}")


def build_selected_file_map(selected_files, original_source_root, file_structure):
    """
    Returns:
      flat   -> {"__flat__": [Path, Path, ...]}
      nested -> {"class_name": [Path, Path, ...], ...}
    """
    source_root = Path(original_source_root)

    if file_structure == "flat":
        return {"__flat__": sorted([Path(p) for p in selected_files])}

    if file_structure == "nested":
        file_map = {}
        for src_path in selected_files:
            src_path = Path(src_path)
            relative_path = src_path.relative_to(source_root)
            class_name = relative_path.parts[0]
            file_map.setdefault(class_name, []).append(src_path)

        for class_name in file_map:
            file_map[class_name] = sorted(file_map[class_name])
        return file_map

    raise ValueError(f"Unknown FILE_STRUCTURE: {file_structure}")


def expected_output_filename(src_path, corruption, intensity):
    src_name = Path(src_path).stem
    return f"{src_name}_{corruption}_{intensity}.JPEG"


def get_missing_files(expected_source_files, target_dir, corruption, intensity):
    target_dir = Path(target_dir)
    if not target_dir.exists():
        return list(expected_source_files)

    existing_names = {
        p.name
        for p in list_existing_outputs_in_dir(target_dir)
        if p.stat().st_size >= MIN_VALID_FILE_SIZE_BYTES
    }

    missing_files = []
    for src_path in expected_source_files:
        expected_name = expected_output_filename(src_path, corruption, intensity)
        if expected_name not in existing_names:
            missing_files.append(src_path)

    return missing_files


def prepare_temp_source_dir(selected_files, original_source_root, keep_relative_structure=True):
    temp_dir = Path(tempfile.mkdtemp(prefix="corruption_src_"))
    original_source_root = Path(original_source_root)

    print(f"Creating temporary source directory at: {temp_dir}")

    for src_path in selected_files:
        src_path = Path(src_path)
        if keep_relative_structure:
            relative_path = src_path.relative_to(original_source_root)
            temp_dest_path = temp_dir / relative_path
        else:
            temp_dest_path = temp_dir / src_path.name
        temp_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, temp_dest_path)

    print(f"Temporary source directory is ready with {len(selected_files)} files.")
    return str(temp_dir)


def run_corruption_process(source_dir, target_dir, corruption, intensity, tile_dir):
    print(
        f"--> Processing: [Corruption: {corruption}, Intensity: {intensity}] "
        f"for {os.path.basename(source_dir)}"
    )

    requires_tiles = corruption in {"mosaic", "sticker"}
    if requires_tiles and not tile_dir:
        print(f"Skipping '{corruption}' because --tile_dir is not set.")
        return False

    command = [
        "python",
        APPLY_SCRIPT_PATH,
        "--source_folder",
        source_dir,
        "--target_folder",
        target_dir,
        "--augment_type",
        corruption,
        "--intensity_level",
        str(intensity),
    ]

    if requires_tiles:
        command.extend(["--tile_dir", tile_dir])

    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        print("      Done.")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"      [ERROR] Failed to process {os.path.basename(source_dir)} "
            f"for {corruption}-{intensity}."
        )
        print(f"      Error output:\n{e.stderr}")
        return False


def run_single_subtask_for_missing_files(
    missing_files,
    source_root,
    target_dir,
    corruption,
    intensity,
    keep_relative_structure,
):
    if not missing_files:
        print(f"      [Skip] Already complete: {target_dir}")
        return "skipped"

    os.makedirs(target_dir, exist_ok=True)
    temp_source_root = None
    try:
        temp_source_root = prepare_temp_source_dir(
            missing_files,
            original_source_root=source_root,
            keep_relative_structure=keep_relative_structure,
        )

        source_path = temp_source_root
        if keep_relative_structure and len(missing_files) > 0:
            first_relative = Path(missing_files[0]).relative_to(Path(source_root))
            if len(first_relative.parts) > 1:
                source_path = os.path.join(temp_source_root, first_relative.parts[0])

        ok = run_corruption_process(
            source_path,
            target_dir,
            corruption,
            intensity,
            TILE_DIR_PATH,
        )
        return "done" if ok else "failed"
    finally:
        if temp_source_root and os.path.exists(temp_source_root):
            print(f"      Cleaning up temporary directory: {temp_source_root}")
            shutil.rmtree(temp_source_root)


def run_single_dataset(cfg):
    print("\n" + "=" * 50)
    print(f"Starting task: {cfg['TASK_NAME']}")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    selected_files = select_files_for_corruption(cfg)
    if not selected_files:
        print("[ERROR] No files were selected for processing. Exiting.")
        return

    selected_file_map = build_selected_file_map(
        selected_files,
        cfg["SOURCE_ROOT"],
        cfg["FILE_STRUCTURE"],
    )

    summary = {"done": 0, "skipped": 0, "failed": 0}

    try:
        if cfg["FILE_STRUCTURE"] == "nested":
            class_folders = sorted(selected_file_map.keys())
            print(
                f"\nFound {len(class_folders)} selected class folders. "
                "Starting corruption process with resume support..."
            )

            for corruption in CORRUPTION_TYPES:
                for intensity in INTENSITY_LEVELS:
                    for class_folder in class_folders:
                        expected_source_files = selected_file_map[class_folder]
                        target_path = os.path.join(
                            cfg["OUTPUT_ROOT"],
                            corruption,
                            f"intensity_level_{intensity}",
                            class_folder,
                        )
                        missing_files = get_missing_files(
                            expected_source_files,
                            target_path,
                            corruption,
                            intensity,
                        )
                        print(
                            f"\n[{class_folder}] {corruption} / intensity={intensity}: "
                            f"{len(expected_source_files) - len(missing_files)}/"
                            f"{len(expected_source_files)} already exist, "
                            f"{len(missing_files)} remaining."
                        )
                        status = run_single_subtask_for_missing_files(
                            missing_files=missing_files,
                            source_root=cfg["SOURCE_ROOT"],
                            target_dir=target_path,
                            corruption=corruption,
                            intensity=intensity,
                            keep_relative_structure=True,
                        )
                        summary[status] += 1

        elif cfg["FILE_STRUCTURE"] == "flat":
            expected_source_files = selected_file_map["__flat__"]
            print("\nProcessing flat directory structure with resume support...")

            for corruption in CORRUPTION_TYPES:
                for intensity in INTENSITY_LEVELS:
                    target_path = os.path.join(
                        cfg["OUTPUT_ROOT"],
                        corruption,
                        f"intensity_level_{intensity}",
                    )
                    missing_files = get_missing_files(
                        expected_source_files,
                        target_path,
                        corruption,
                        intensity,
                    )
                    print(
                        f"\n{corruption} / intensity={intensity}: "
                        f"{len(expected_source_files) - len(missing_files)}/"
                        f"{len(expected_source_files)} already exist, "
                        f"{len(missing_files)} remaining."
                    )
                    status = run_single_subtask_for_missing_files(
                        missing_files=missing_files,
                        source_root=cfg["SOURCE_ROOT"],
                        target_dir=target_path,
                        corruption=corruption,
                        intensity=intensity,
                        keep_relative_structure=False,
                    )
                    summary[status] += 1

        else:
            raise ValueError(f"Unknown FILE_STRUCTURE: {cfg['FILE_STRUCTURE']}")

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        raise

    print(
        f"\nTask completed: {cfg['TASK_NAME']} | "
        f"done={summary['done']}, skipped={summary['skipped']}, failed={summary['failed']}"
    )


def main():
    if not ENABLED_DATASETS:
        raise ValueError("ENABLED_DATASETS is empty.")

    for dataset_name in ENABLED_DATASETS:
        if dataset_name not in DATASETS:
            raise KeyError(f"Unknown dataset key in ENABLED_DATASETS: {dataset_name}")

    print("--- Starting merged OOD corruption generation script (resume enabled, robust naming only) ---")
    print("Enabled datasets:")
    for dataset_name in ENABLED_DATASETS:
        print(f"  - {dataset_name}")

    for dataset_name in ENABLED_DATASETS:
        run_single_dataset(DATASETS[dataset_name])

    print("\nAll enabled tasks completed!")


if __name__ == "__main__":
    main()
