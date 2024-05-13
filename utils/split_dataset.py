import json
import os
import random
import shutil
import tempfile

import fire


class SplitDataset:
    def __init__(self, split_ratio: str, path: str):
        self.split_ratio = json.loads(split_ratio)
        self.path = path

        if not self.verify_split_ratio():
            raise ValueError("Split ratio must sum to 1 or less")
        if not self.verify_dataset():
            raise ValueError("Dataset path must contain directories")

    def verify_split_ratio(self) -> bool:
        sum = 0
        for _, ratio in self.split_ratio.items():
            sum += ratio
            if sum > 1:
                return False

        return True

    def verify_dataset(self) -> bool:
        labels = os.listdir(self.path)

        for label in labels:
            path = os.path.join(self.path, label)
            if not os.path.isdir(path):
                return False

        return True

    def temp_copy(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_name = self.temp_dir.name
        print(f"Create temp directory: {temp_name}")

        self.temp_path = f"{temp_name}/dataset"

        print("Copying dataset to temp directory")
        shutil.copytree(self.path, self.temp_path)
        print("Copy finished successfully")

    def split(self, shuffle: bool = True) -> dict:
        labels = os.listdir(self.path)
        print(f"Labels: {labels}")

        if not hasattr(self, "temp_path"):
            self.temp_copy()

        self.split_dataset = dict()
        start_indexes = {label: 0 for label in labels}
        label_elements = dict()
        element_length = dict()

        # Cache dataset info
        for label in labels:
            elements = os.listdir(f"{self.temp_path}/{label}")
            length = len(elements)

            label_elements[label] = elements
            element_length[label] = length

            print(f"{label}: {length}")

            if shuffle:
                random.shuffle(elements)

        for split_type, ratio in self.split_ratio.items():
            split_items = dict()
            print(f"--- Split type: {split_type} ---\n")
            for label in labels:
                elements = label_elements[label]
                length = element_length[label]
                start_index = start_indexes[label]

                split_count = int(length * ratio)
                split_items[label] = elements[start_index : start_index + split_count]
                start_indexes[label] += split_count

                print(f"{label}: {split_count}")

            print(f"\n--- Split type: {split_type} finish ---\n")
            self.split_dataset[split_type] = split_items

        return self.split_dataset

    def save_split(self, dst: str) -> dict:
        output_path = dict()

        split_path = f"{dst}/split_dataset"
        if not os.path.exists(split_path):
            os.makedirs(split_path, exist_ok=True)

        for dataset_type, _ in self.split_dataset.items():
            type_path = f"{split_path}/{dataset_type}"
            if not os.path.exists(type_path):
                print(f"Create directory: {type_path}")
                os.mkdir(type_path)

            output_path[dataset_type] = type_path

        for dataset_type, dataset in self.split_dataset.items():
            for label, _ in dataset.items():
                label_path = f"{split_path}/{dataset_type}/{label}"
                if not os.path.exists(label_path):
                    print(f"Create directory: {label_path}")
                    os.mkdir(label_path)

        for dataset_type, dataset in self.split_dataset.items():
            for label, elements in dataset.items():
                for element in elements:
                    src_file = f"{self.temp_path}/{label}/{element}"
                    dst_file = f"{split_path}/{dataset_type}/{label}/{element}"
                    shutil.copy(src_file, dst_file)

        self.temp_dir.cleanup()

        print(f"Split dataset saved at {split_path}")
        return output_path


def main(
    dataset_path="dataset",
    split_output="/tmp/split_output",
    split_ratio='{"train":0.725,"validation":0.225,"test":0.05}',
):
    dataset = SplitDataset(split_ratio, dataset_path)
    dataset.split()
    dataset.save_split(split_output)


if __name__ == "__main__":
    fire.Fire(main)
