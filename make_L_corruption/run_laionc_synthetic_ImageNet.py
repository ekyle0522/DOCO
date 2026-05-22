import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/d/stamp_lib/datasets"))

CONFIG = {
    "TASK_NAME": "ImageNet-C-Subset (resume with source-aligned filenames)",
    "SOURCE_ROOT": str(DATA_ROOT / "ImageNet" / "val"),
    "OUTPUT_ROOT": str(DATA_ROOT / "ImageNet-LAION-5K"),
    "FILE_STRUCTURE": "nested",
    "SAMPLING_MODE": "from_file",
    "SAMPLE_LIST_FILE": str(REPO_ROOT / "imagenet" / "robustbench" / "data" / "imagenet_test_image_ids_5k.txt"),
    "NUM_SAMPLES": 5000,
}


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


def is_supported_source_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def select_files_for_corruption(cfg):
    print(f"Selecting {cfg['NUM_SAMPLES']} files using '{cfg['SAMPLING_MODE']}' mode...")

    source_root = Path(cfg["SOURCE_ROOT"])
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    if cfg["SAMPLING_MODE"] != "from_file":
        raise ValueError("This script expects SAMPLING_MODE='from_file'.")

    sample_list_path = Path(cfg["SAMPLE_LIST_FILE"])
    if not sample_list_path.is_file():
        raise FileNotFoundError(f"Sample list file not found: {sample_list_path}")

    print(f"Reading sample list from: {sample_list_path}")
    with open(sample_list_path, "r") as f:
        relative_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(relative_paths)} total entries in the list file.")
    selected_relative_paths = relative_paths[: cfg["NUM_SAMPLES"]]
    selected_files = [source_root / rel_path for rel_path in selected_relative_paths]

    missing_sources = [p for p in selected_files if not p.exists()]
    if missing_sources:
        raise FileNotFoundError(
            f"Some selected source files do not exist. Example: {missing_sources[0]}"
        )

    unsupported = [p for p in selected_files if not is_supported_source_image(p)]
    if unsupported:
        print(
            f"[Warning] {len(unsupported)} selected files are not .jpg/.jpeg and will be ignored. "
            f"Example: {unsupported[0]}"
        )

    selected_files = [p for p in selected_files if is_supported_source_image(p)]
    print(f"Selected {len(selected_files)} supported files based on the provided list.")
    return selected_files


def build_selected_file_map(selected_files, original_source_root):
    source_root = Path(original_source_root)
    file_map = {}
    for src_path in selected_files:
        relative_path = Path(src_path).relative_to(source_root)
        class_name = relative_path.parts[0]
        file_map.setdefault(class_name, []).append(Path(src_path))

    for class_name in file_map:
        file_map[class_name] = sorted(file_map[class_name])
    return file_map


def expected_output_filename(src_path: Path) -> str:
    return src_path.name


def get_missing_files(expected_source_files, target_dir):
    target_dir = Path(target_dir)
    if not target_dir.exists():
        return list(expected_source_files)

    existing_names = {
        p.name
        for p in target_dir.iterdir()
        if p.is_file() and p.stat().st_size >= MIN_VALID_FILE_SIZE_BYTES
    }

    missing_files = []
    for src_path in expected_source_files:
        expected_name = expected_output_filename(src_path)
        if expected_name not in existing_names:
            missing_files.append(src_path)
    return missing_files


def prepare_temp_source_dir(selected_files, original_source_root):
    temp_dir = Path(tempfile.mkdtemp(prefix="corruption_src_"))
    original_source_root = Path(original_source_root)

    print(f"Creating temporary source directory at: {temp_dir}")
    for src_path in selected_files:
        src_path = Path(src_path)
        relative_path = src_path.relative_to(original_source_root)
        temp_dest_path = temp_dir / relative_path
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
        "--keep_original_name",
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


def run_single_subtask_for_missing_files(missing_files, source_root, target_dir, corruption, intensity, class_folder):
    if not missing_files:
        print(f"      [Skip] Already complete: {target_dir}")
        return "skipped"

    os.makedirs(target_dir, exist_ok=True)
    temp_source_root = None
    try:
        temp_source_root = prepare_temp_source_dir(
            missing_files,
            original_source_root=source_root,
        )
        source_path = os.path.join(temp_source_root, class_folder)
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


def main():
    print("--- Starting ImageNet corruption generation script (resume with source-aligned filenames) ---")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------------------------------------------")

    selected_files = select_files_for_corruption(CONFIG)
    if not selected_files:
        print("[ERROR] No files were selected for processing. Exiting.")
        return

    selected_file_map = build_selected_file_map(selected_files, CONFIG["SOURCE_ROOT"])
    class_folders = sorted(selected_file_map.keys())
    print(
        f"\nFound {len(class_folders)} selected class folders. "
        "Starting corruption process with resume support..."
    )

    summary = {"done": 0, "skipped": 0, "failed": 0}

    for corruption in CORRUPTION_TYPES:
        for intensity in INTENSITY_LEVELS:
            for class_folder in class_folders:
                expected_source_files = selected_file_map[class_folder]
                target_path = os.path.join(
                    CONFIG["OUTPUT_ROOT"],
                    corruption,
                    f"intensity_level_{intensity}",
                    class_folder,
                )
                missing_files = get_missing_files(expected_source_files, target_path)
                print(
                    f"\n[{class_folder}] {corruption} / intensity={intensity}: "
                    f"{len(expected_source_files) - len(missing_files)}/"
                    f"{len(expected_source_files)} already exist, "
                    f"{len(missing_files)} remaining."
                )
                status = run_single_subtask_for_missing_files(
                    missing_files=missing_files,
                    source_root=CONFIG["SOURCE_ROOT"],
                    target_dir=target_path,
                    corruption=corruption,
                    intensity=intensity,
                    class_folder=class_folder,
                )
                summary[status] += 1

    print(
        f"\nTask completed: {CONFIG['TASK_NAME']} | "
        f"done={summary['done']}, skipped={summary['skipped']}, failed={summary['failed']}"
    )
    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()
