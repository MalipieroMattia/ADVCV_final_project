from sklearn.model_selection import train_test_split
import os
import shutil


# need to take all files from the 2 raw yolo folders and merge them in one folder not split into train and val
def unsplit_dataset(split_data_dir, output_dir):
    """
    Merges train/val split folders back into single images and labels folders.

    Args:
        split_data_dir: Path to directory containing images/{train,val} and labels/{train,val}
        output_dir: Path where merged images/ and labels/ folders will be created
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    # Iterate through train and val subsets
    for subset in ["train", "val"]:
        images_subset_dir = f"{split_data_dir}/images/{subset}"
        labels_subset_dir = f"{split_data_dir}/labels/{subset}"

        # Check if subset directories exist
        if not os.path.exists(images_subset_dir):
            continue

        # Copy all images from this subset
        for image in os.listdir(images_subset_dir):
            if image.endswith(".jpg"):
                shutil.copy(
                    f"{images_subset_dir}/{image}", f"{output_dir}/images/{image}"
                )

                # Copy corresponding label file
                label_file = image.replace(".jpg", ".txt")
                label_path = f"{labels_subset_dir}/{label_file}"
                if os.path.exists(label_path):
                    shutil.copy(label_path, f"{output_dir}/labels/{label_file}")

    print(f"Unsplit complete! Merged data saved to {output_dir}")


def split_dataset(
    images_dir,
    labels_dir,
    output_dir,
    test_size=0.2,
    val_size=0.2,
    is_already_split=False,
):
    """
    Splits dataset into train/val/test sets.

    Args:
        images_dir: Path to images (if is_already_split=True, path to root with images/{train,val})
        labels_dir: Path to labels (if is_already_split=True, path to root with labels/{train,val})
        output_dir: Path where split dataset will be saved
        test_size: Proportion for test set (default 0.2)
        val_size: Proportion for validation set from remaining data (default 0.2)
        is_already_split: If True, will first unsplit the data before re-splitting
    """

    if is_already_split:
        temp_unsplit_dir = f"{output_dir}_temp_unsplit"
        unsplit_dataset(images_dir, temp_unsplit_dir)
        images_dir = f"{temp_unsplit_dir}/images"
        labels_dir = f"{temp_unsplit_dir}/labels"

    images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    train_images, test_images = train_test_split(
        images, test_size=test_size, random_state=42
    )
    train_images, val_images = train_test_split(
        train_images, test_size=val_size, random_state=42
    )

    for subset, subset_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        os.makedirs(f"{output_dir}/images/{subset}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{subset}", exist_ok=True)
        for image in subset_images:
            shutil.copy(
                f"{images_dir}/{image}", f"{output_dir}/images/{subset}/{image}"
            )
            label_file = image.replace(".jpg", ".txt")
            shutil.copy(
                f"{labels_dir}/{label_file}",
                f"{output_dir}/labels/{subset}/{label_file}",
            )

    # Clean up temporary unsplit directory if it was created
    if is_already_split:
        temp_unsplit_dir = f"{output_dir}_temp_unsplit"
        if os.path.exists(temp_unsplit_dir):
            shutil.rmtree(temp_unsplit_dir)

    print(
        f" Dataset saved to {output_dir}, Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}"
    )
