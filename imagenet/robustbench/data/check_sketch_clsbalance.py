import pandas as pd
from pathlib import Path

def check_class_balance(file_path, num_samples=5000):
    """
    Check class balance in the first N samples of a list file.

    Args:
        file_path (str): Path to the sample list file.
        num_samples (int): Number of leading samples to inspect.
    """
    try:
        # Read the first num_samples rows.
        # Expected format: "path class_label", space-separated with no header.
        df = pd.read_csv(
            file_path,
            sep=' ',
            header=None,
            names=['path', 'class_label'],
            nrows=num_samples
        )

        # Warn if the file has fewer rows than requested.
        if len(df) < num_samples:
            print(f"Warning: file has fewer than {num_samples} rows; read {len(df)} rows.")

        # Count occurrences in the class_label column.
        class_counts = df['class_label'].value_counts()

        print(f"--- Class distribution for the first {len(df)} samples ---")
        print("Samples per class:")
        print(class_counts)

        # Analyze balance.
        print("\n--- Balance analysis ---")
        if class_counts.nunique() == 1:
            print(f"All classes have exactly {class_counts.iloc[0]} samples; the subset is balanced.")
        else:
            min_count = class_counts.min()
            max_count = class_counts.max()
            print("Class sample counts are not identical.")
            print(f"Minimum samples per class: {min_count}")
            print(f"Maximum samples per class: {max_count}")
            # ImageNet-Sketch has 1000 classes.
            # If balanced, 5000 samples should give 5000 / 1000 = 5 samples per class.
            expected_count = num_samples / 1000
            print(f"For ImageNet-Sketch (1000 classes), the ideal count is {expected_count} samples per class.")
            if min_count == max_count:
                 print("All observed classes have the same number of samples; observed classes are balanced.")
            else:
                 print("Sample counts differ across classes; this is not strictly class-balanced.")


    except FileNotFoundError:
        print(f"Error: file not found: '{file_path}'. Please check the path.")
    except Exception as e:
        print(f"Error while processing file: {e}")

# --- Main ---
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    file_path = data_dir / 'sketchPath.txt'
    check_class_balance(file_path, num_samples=5000)
