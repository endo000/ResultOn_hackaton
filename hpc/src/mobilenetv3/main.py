import json
import os
import random
import shutil
import tempfile

import fire
import mlflow
import mlflow.keras
import numpy as np
import pika
import tensorflow as tf
import tensorflow_hub as hub


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


class RabbitMQ:
    def __init__(self, exchange="hackathon", routing_key="mobilenetv3"):
        self.exchange = exchange
        self.routing_key = routing_key

        credentials = pika.PlainCredentials(
            os.getenv("PIKA_USER", "guest"),
            os.getenv("PIKA_PASS", "guest"),
        )
        parameters = pika.ConnectionParameters(
            host=os.getenv("PIKA_HOST", "localhost"),
            port=os.getenv("PIKA_PORT", "5672"),
            credentials=credentials,
        )
        connection = pika.BlockingConnection(parameters)

        self.channel = connection.channel()
        self.channel.exchange_declare(exchange=self.exchange, exchange_type="topic")

    def publish(self, message):
        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=self.routing_key,
            body=message,
        )


class MobileNetV3Train:
    def __init__(
        self,
        model_name,
        variation,
        dataset,
        output,
        epochs,
        augment,
        normalize,
        softmax,
        batch_size,
        img_width,
        img_height,
        fine_tune=True,
        test_size=30,
    ):
        self.model_name = model_name
        self.variation = variation
        self.dataset = dataset
        self.output = output
        self.epochs = epochs
        self.augment = augment
        self.normalize = normalize
        self.softmax = softmax
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.shape = (img_width, img_height, 3)
        self.fine_tune = fine_tune
        self.test_size = test_size

        self.model_url = (
            "https://www.kaggle.com/models/google/mobilenet-v3/TensorFlow2/"
            f"{variation}/1"
        )

        self.rabbit = RabbitMQ()

    def keras_dataset(self, path, dataset_type, **kwargs):
        dataset = tf.keras.utils.image_dataset_from_directory(
            f"{path}/{dataset_type}",
            color_mode="rgb",
            image_size=(self.img_height, self.img_width),
            **kwargs,
        )
        return dataset

    def keras_datasets(self):
        self.train = self.keras_dataset(self.dataset, "train")
        self.validation = self.keras_dataset(self.dataset, "validation")
        self.test = self.keras_dataset(self.dataset, "test", batch_size=1)

        self.class_names = self.train.class_names
        print(f"Class names: {self.class_names}")

        if self.normalize:
            norm_layer = tf.keras.layers.Rescaling(1.0 / 255)
            self.train = self.train.map(lambda x, y: (norm_layer(x), y))
            self.validation = self.validation.map(lambda x, y: (norm_layer(x), y))
            self.test = self.test.map(lambda x, y: (norm_layer(x), y[0]))
        else:
            self.test = self.test.map(lambda x, y: (x, y[0]))

        self.train = (
            self.train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        self.validation = self.validation.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def full_name(self):
        name = f"{self.model_name}_{self.variation}"
        # if self.augment:
        #     name += "_augment-fixed"
        # if self.normalize:
        #     name += "_normalize"
        # if self.softmax:
        #     name += "_softmax"
        # tensorboard_model_name = (
        #         "mobilenet/"
        #         + name
        #         + "/"
        #         + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # )
        return name

    def build_model(self):
        self.rabbit.publish("Building new model")

        self.model = tf.keras.Sequential(
            [
                # Explicitly define the input shape so the model can be properly
                # loaded by the TFLiteConverter
                tf.keras.layers.InputLayer(input_shape=self.shape),
                *(
                    (tf.keras.layers.Rescaling(1.0 / 255, input_shape=self.shape),)
                    if not self.normalize
                    else ()
                ),
                hub.KerasLayer(self.model_url, trainable=self.fine_tune),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1024, activation="relu", name="hidden_layer"),
                tf.keras.layers.Dense(len(self.class_names), name="output"),
            ]
        )

        # tf.keras.layers.Dense(
        #     class_len, kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        # ),

        self.model.summary(expand_nested=True)
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return self.model

    def fit_model(self):
        self.rabbit.publish("Fit new model")

        mlflow.set_experiment("mobilenetv3")
        mlflow.tensorflow.autolog()
        with mlflow.start_run():
            mlflow.log_param("classes", self.class_names)
            history = self.model.fit(
                self.train,
                validation_data=self.validation,
                epochs=self.epochs,
            )
            if self.softmax:
                self.model = tf.keras.Sequential(
                    [self.model, tf.keras.layers.Softmax()]
                )

            mlflow.keras.log_model(
                self.model,
                "model",
                registered_model_name=self.full_name(),
                metadata={"classes": self.class_names},
            )

        return self.model, history

    def save_model(self):
        model_path = f"{self.output}/{'_'.join(self.class_names)}/{self.full_name()}"
        print(f"Saving TF model to {model_path}")
        self.model.save(model_path)

    def run_test(self):
        correct = 0
        print("\n--- Running test ---\n")

        for i, (image, label) in enumerate(self.test.take(self.test_size)):
            predictions = self.model.predict(image)

            if np.argmax(predictions) == label:
                correct += 1
            print(
                f"{self.class_names[label]},",
                f"{predictions} -",
                f"{'correct' if np.argmax(predictions) == label else 'incorrect'}",
                sep=" ",
            )

        print(
            f"\n--- Accuracy: {correct / self.test_size * 100}%",
            f"{correct}/{self.test_size} ---",
            sep=" ",
        )
        print("--- Test finished ---")

    def train(self):
        self.keras_datasets()
        self.build_model()
        self.fit_model()
        self.run_test()


def available_variations():
    print(
        "large-100-224-classification",
        "large-075-224-classification",
        "small-100-224-classification",
        "small-075-224-classification",
        sep="\n",
    )


def split_dataset(
    dataset_path="dataset",
    split_output="/tmp/split_output",
    split_ratio='{"train":0.725,"validation":0.225,"test":0.05}',
):
    dataset = SplitDataset(split_ratio, dataset_path)
    dataset.split()
    dataset.save_split(split_output)


def train(
    model_name="mobilenetv3",
    variation="small-100-224-classification",
    dataset_path="/tmp/split_output",
    output_path="/tmp/models",
    epochs=20,
    augment=False,
    normalize=False,
    softmax=False,
    batch_size=32,
    img_width=224,
    img_height=224,
):
    mobilenet = MobileNetV3Train(
        model_name,
        variation,
        dataset_path,
        output_path,
        epochs,
        augment,
        normalize,
        softmax,
        batch_size,
        img_width,
        img_height,
    )
    mobilenet.train()


def test_gpu():
    print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    print(f"Build info: {tf.sysconfig.get_build_info()}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


if __name__ == "__main__":
    random.seed(123)
    fire.Fire(
        {
            "available_variations": available_variations,
            "split_dataset": split_dataset,
            "train": train,
            "test_gpu": test_gpu,
        }
    )
