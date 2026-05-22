import os
from pathlib import Path

def compare_files(file1_path, file2_path):
    """
    Compare how many leading lines are identical between two files.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.
    """
    if not os.path.exists(file1_path):
        print(f"Error: file not found -> {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Error: file not found -> {file2_path}")
        return

    identical_lines_count = 0
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            # zip stops automatically when the shorter file ends.
            for line1, line2 in zip(f1, f2):
                # strip() removes trailing whitespace so line endings do not affect comparison.
                if line1.strip() == line2.strip():
                    identical_lines_count += 1
                else:
                    # Stop at the first mismatch.
                    break
        
        print(
            f"Files '{os.path.basename(file1_path)}' and "
            f"'{os.path.basename(file2_path)}' have {identical_lines_count} "
            "identical leading lines."
        )

    except Exception as e:
        print(f"Error while reading files: {e}")


# --- Update these paths if needed. ---
data_dir = Path(__file__).resolve().parent
# 5k sample list.
file_5k_path = data_dir / 'imagenet_test_image_ids_5k.txt'
# 50k sample list.
file_50k_path = data_dir / 'imagenet_test_image_ids.txt'

# Run comparison.
compare_files(file_5k_path, file_50k_path)
