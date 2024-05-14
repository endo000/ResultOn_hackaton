import os
import random

import boto3
import fire
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from botocore.client import Config
from rabbitmq import RabbitMQ


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

        self.rabbit = RabbitMQ(routing_key="mobilenetv3")

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
        name = f"{self.model_name}"
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
        self.rabbit.publish("Building model...")

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

        self.rabbit.publish("Compiling model...")

        self.model.summary(expand_nested=True)
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return self.model

    def fit_model(self):
        self.rabbit.publish("Training model...")

        mlflow.set_experiment("mobilenetv3")
        mlflow.tensorflow.autolog()
        with mlflow.start_run() as run:
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

            self.rabbit.publish("Training finished successfully...")

            self.save_model_s3()

            mlflow.keras.log_model(
                self.model,
                "model",
                registered_model_name=self.full_name(),
                metadata={"classes": self.class_names},
            )

        self.rabbit.publish(f"Model name: {self.full_name()}")

        return self.model, history

    def save_model_s3(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open("model", "wb") as f:
            f.write(tflite_model)

        s3 = boto3.client('s3',
                    endpoint_url=os.environ['MINIO_URI'],
                    aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
                    aws_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY'],
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

        with open("model.tflite", "rb") as f:
            s3.upload_fileobj(f, 'mlflow', f"{self.full_name()}/1/model.tflite")

    def save_model(self):
        self.model_path = "model"
        print(f"Saving TF model to {self.model_path}")
        self.model.save(self.model_path)
        return self.model_path

    def run_test(self):
        self.rabbit.publish("Testing model...")

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

        accuracy = f"{correct / self.test_size * 100}%"
        self.rabbit.publish(f"Testing finished, accuracy: {accuracy}.")

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
        self.save_model()
        self.run_test()


def available_variations():
    print(
        "large-100-224-classification",
        "large-075-224-classification",
        "small-100-224-classification",
        "small-075-224-classification",
        sep="\n",
    )


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
            "train": train,
            "test_gpu": test_gpu,
        }
    )
