import os
import json

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from config import IMAGE_SIZE
from utils.cnn_model import build_model
from utils.database import collection
from utils.preprocess import preprocess_image


MODEL_PATH = os.path.join("model", "face_model.h5")
LABELS_PATH = os.path.join("model", "face_labels.json")


def load_training_data():
    images = []
    labels = []

    for person in collection.find({}, {"name": 1, "images": 1}):
        person_name = str(person.get("name", "")).strip()
        if not person_name:
            continue
        for image_binary in person.get("images", []):
            image = preprocess_image(bytes(image_binary))
            if image is not None:
                images.append(image)
                labels.append(person_name)

    if not images:
        raise RuntimeError(
            "No usable images found in MongoDB. Register at least one person "
            "with a face image before training."
        )

    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=str)


def main():
    images, labels = load_training_data()
    class_labels = sorted(set(labels.tolist()))
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    encoded_labels = np.asarray(
        [label_to_index[label] for label in labels], dtype=np.int32
    )

    model = build_model(len(class_labels))
    fit_kwargs = {"epochs": 80, "batch_size": 8, "verbose": 1, "shuffle": True}
    if len(images) >= 40:
        fit_kwargs["validation_split"] = 0.2
        fit_kwargs["callbacks"] = [EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )]

    model.fit(images, encoded_labels, **fit_kwargs)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as labels_file:
        json.dump(class_labels, labels_file)
    print(f"Saved model for {len(class_labels)} people to {MODEL_PATH}")


if __name__ == "__main__":
    main()