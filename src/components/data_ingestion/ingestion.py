import os
import re
import shutil

from tqdm import tqdm


class DataIngestion:
    """Handles organizing data by transferring files to specified directories and renaming them systematically for further processing."""

    def move_images_to_directories(
        self, data_dir: str, input_images_dir: str, target_images_dir: str
    ) -> None:
        """
        Moves images from a raw data directory to their respective directories.

        Args:
            data_dir (str): Directory where all the images are stored
            input_images_dir (str): Directory where input images are stored
            target_images_dir (str): Directory where target images are stored
        """
        dirs = [_[0] for _ in os.walk(data_dir)][1:]

        for patient_path in tqdm(dirs):
            images = os.listdir(patient_path)
            images.sort(key=lambda _: int(re.sub("\D", "", _)))

            for image_path in images:
                if "_mask" not in image_path:
                    shutil.copy(
                        os.path.join(patient_path, image_path),
                        os.path.join(input_images_dir, image_path),
                    )
                else:
                    shutil.copy(
                        os.path.join(patient_path, image_path),
                        os.path.join(target_images_dir, image_path),
                    )

    def rename_images_by_index(
        self, input_images_dir: str, target_images_dir: str
    ) -> None:
        """
        Renames image path names by their index.

        Args:
            input_images_dir (str): Directory where input images are stored
            target_images_dir (str): Directory where target images are stored
        """
        images_list = os.listdir(input_images_dir)
        masked_images_list = os.listdir(target_images_dir)

        images_list.sort(key=lambda _: int(re.sub("\D", "", _)))
        masked_images_list.sort(key=lambda _: int(re.sub("\D", "", _)))

        for idx, image_path in tqdm(enumerate(images_list)):
            os.replace(
                os.path.join(input_images_dir, image_path),
                os.path.join(input_images_dir, f"image_{idx}.tif"),
            )

        for idx, image_path in tqdm(enumerate(masked_images_list)):
            os.replace(
                os.path.join(target_images_dir, image_path),
                os.path.join(target_images_dir, f"image_{idx}.tif"),
            )
