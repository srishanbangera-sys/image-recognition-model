from flask import Flask, render_template, request
from bson.binary import Binary
import numpy as np
import os

from utils.database import collection
from utils.preprocess import preprocess_image
from config import IMAGE_SIZE


app = Flask(__name__)


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

    label = collection.count_documents({})
    image_binary = [Binary(file.read()) for file in files]

    person = {
        "name" : name, 
        "age" : age,
        "job" : job,
        "phhone" : phone,
        "address" : address,
        "label" : label,
        "images" : image_binary
    }


    collection.insert_one(person)

    return "Person Registered Sucessfully"

@app.route("/recognize")
def recognize_page():
    return render_template("recognize.html")


@app.route("/predict", methods = ["POST"])
def predict():
    model_path = "model/face_model.h5"

    if not os.path.exists(model_path):
        return "Model not trained yet"

    from tensorflow.keras.model import load_model
    




