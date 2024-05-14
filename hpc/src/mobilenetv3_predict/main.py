import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


def prepare_img(image_path, target_size, color_mode="rgb"):
    img = tf.keras.utils.load_img(
        image_path, target_size=target_size, color_mode=color_mode
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    return img_array


def get_output(predictions, class_names=None):
    score = tf.nn.softmax(predictions[0])

    if class_names is None:
        class_names = [i for i in range(len(score))]

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )


if __name__ == "__main__":
    # model = mlflow.tensorflow.load_model("models:/contracts_small-100-224-classification/2")
    model = tf.keras.models.load_model("/exec/models/car_dashboard_documents/contracts_small-100-224-classification")

    img_array = prepare_img(
        "/exec/cat.jpg",
        (224, 224),
    )

    predictions = model.predict(img_array)
