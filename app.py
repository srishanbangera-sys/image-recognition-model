from flask import Flask, render_template, request
from bson.binary import Binary
import json
import numpy as np
import os

from utils.database import collection
from utils.preprocess import preprocess_image
app = Flask(__name__)
MODEL_PATH = os.path.join(app.root_path, "model", "face_model.h5")
LABELS_PATH = os.path.join(app.root_path, "model", "face_labels.json")


@app.route("/")
def home():
    return render_template("register.html")


@app.route("/register", methods = ["POST"])
def register():

    name = request.form["name"]
    age = request.form["age"]
    job = request.form["job"]
    phone = request.form["phone"]
    address = request.form["address"]

    files = request.files.getlist("photos")

    last_person = collection.find_one(sort=[("label", -1)])
    label = int(last_person.get("label", 0)) + 1 if last_person else 1
    image_binary = [Binary(file.read()) for file in files]

    if not files or any(not file.filename for file in files):
        return "At least one image is required", 400

    person = {
        "name" : name, 
        "age" : age,
        "job" : job,
        "phone" : phone,
        "address" : address,
        "label" : label,
        "images" : image_binary
    }


    collection.insert_one(person)

    return "Person Registered Sucessfully"

@app.route("/recognize")
def recognize_page():
    return render_template("recognise.html")


@app.route("/predict", methods = ["POST"])
def predict():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return "Model not trained yet"

    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)

    file = request.files.get("photo")
    if file is None or not file.filename:
        return "An image is required", 400
    img = preprocess_image(file.read())
    if img is None:
        return "The uploaded file is not a valid image", 400
    img = np.expand_dims(img, axis = 0)

    prediction = model.predict(img)

    with open(LABELS_PATH, encoding="utf-8") as labels_file:
        class_labels = json.load(labels_file)

    predicted_index = int(np.argmax(prediction[0]))
    if predicted_index >= len(class_labels):
        return "The model labels are invalid", 500
    person = collection.find_one({"name": class_labels[predicted_index]})

    if person:
        return render_template("result.html", person = person)

    else :
        return "Unknown Person"

if __name__ == "__main__":
    app.run(debug = True)






