import os
from pathlib import Path

import webdataset as wds
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/d/stamp_lib/datasets"))
SOURCE_IMAGE_ROOT = str(DATA_ROOT / "ImageNet" / "val")
OUTPUT_DIR = str(DATA_ROOT)
OUTPUT_FILENAME = "imagenet_val_tiles.tar"


def create_webdataset_from_folder(source_path, output_tar_path):
    source_root = Path(source_path)
    output_path = Path(output_tar_path)

    print("--- WebDataset creation script ---")
    if not source_root.is_dir():
        print(f"[ERROR] Source directory not found: {source_root}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source path: {source_root}")
    print(f"Output file: {output_path}")
    print("\n[1/3] Scanning image files...")

    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    all_image_paths = sorted(
        list(set(p for ext in image_extensions for p in source_root.rglob(f"*{ext}")))
    )

    if not all_image_paths:
        print("[ERROR] No image files were found in the source directory.")
        return

    print(f"Found {len(all_image_paths)} images.")
    print("\n[2/3] Writing images and labels to the tar file...")

    count = 0
    try:
        with wds.TarWriter(str(output_path)) as sink:
            for image_path in tqdm(all_image_paths, desc="Packing"):
                class_name = image_path.parent.name
                relative_path = image_path.relative_to(source_root)
                key_str = str(relative_path).replace(os.sep, "_").rsplit(".", 1)[0]

                with open(image_path, "rb") as stream:
                    image_bytes = stream.read()

                # Keep `cls.txt` to force text handling and match the loader.
                sample = {
                    "__key__": key_str,
                    "jpg": image_bytes,
                    "cls.txt": class_name.encode("utf-8"),
                }
                sink.write(sample)
                count += 1
    except Exception as e:
        print(f"\n[ERROR] Failed while creating the tar file: {e}")
        return

    print("\n[3/3] Done.")
    print(f"Successfully wrote {count} samples to {output_path}")


if __name__ == "__main__":
    output_file_full_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if os.path.exists(output_file_full_path):
        print(f"Removing existing tar file: {output_file_full_path}")
        os.remove(output_file_full_path)

    create_webdataset_from_folder(SOURCE_IMAGE_ROOT, output_file_full_path)
