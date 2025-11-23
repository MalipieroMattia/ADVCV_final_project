"""
Script to merge train/val splits into a single combined dataset for EDA
"""

from utils.data_loader import unsplit_dataset
import os

if __name__ == "__main__":
    # Source: current split data
    split_data_dir = "raw_data/Data_YOLO"

    # Destination: combined data for EDA
    output_dir = "data/combined_dataset"

    print("=" * 60)
    print("UNSPLITTING DATASET FOR EDA")
    print("=" * 60)
    print(f"\nSource: {split_data_dir}")
    print(f"Output: {output_dir}")

    # Create the combined dataset
    unsplit_dataset(split_data_dir, output_dir)

    # Verify the results
    images_count = len(os.listdir(f"{output_dir}/images"))
    labels_count = len(os.listdir(f"{output_dir}/labels"))

    print(f"\n✅ Combined dataset created!")
    print(f"   - Images: {images_count}")
    print(f"   - Labels: {labels_count}")
    print(f"\nYou can now run EDA on: {output_dir}")
