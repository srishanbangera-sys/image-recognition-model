# Face Recognition Web Application

A Flask web application that registers people and their face images in MongoDB, trains a TensorFlow/Keras convolutional neural network, and predicts which registered person appears in a new image.

The application workflow is:

```text
Register person and images -> Store in MongoDB -> Train CNN -> Upload image -> Predict person
```

## Features

- Register a person's name, age, job, phone number, address, and one or more images.
- Store person records and image bytes in MongoDB.
- Resize and normalize images consistently for training and prediction.
- Train a CNN using images stored in MongoDB.
- Recognize a person through a browser upload form.
- Display the matched person's details.
- Keep a generated class map in `model/face_labels.json` so model outputs map to names correctly.

## Technology Stack

- Python 3.10+
- Flask
- MongoDB and PyMongo
- TensorFlow and Keras
- OpenCV
- NumPy
- Jinja2 templates

## Requirements

Install Python dependencies in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install flask pymongo bson numpy opencv-python tensorflow
```

On Windows, activate the environment with:

```powershell
venv\Scripts\activate
```

TensorFlow may require additional platform-specific installation steps. CPU training is supported; a CUDA-capable GPU is optional.

## MongoDB Setup

MongoDB must be running locally before starting the application. The default configuration is:

```python
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "faceDB"
COLLECTION_NAME = "persons"
```

These values are defined in `config.py`. Change them if MongoDB is running at a different address.

Each registered person is stored approximately as:

```json
{
	"name": "Person name",
	"age": "30",
	"job": "Job title",
	"phone": "Phone number",
	"address": "Address",
	"label": 1,
	"images": ["binary image data"]
}
```

The model trains classes by `name`, not by the numeric `label`. This prevents old or duplicate numeric labels from causing incorrect person lookups.

## Run the Application

1. Start MongoDB.
2. Activate the Python virtual environment.
3. Start Flask:

```bash
python app.py
```

Open the application at:

```text
http://127.0.0.1:5000/
```

## Register People

Open the home page and submit:

- Name
- Age
- Job
- Phone
- Address
- One or more face images

Use several clear images for each person. Images should show the person's face at different angles and under slightly different lighting conditions.

## Train the Model

After registering people, run:

```bash
python train_model.py
```

The script:

1. Reads all registered images from MongoDB.
2. Decodes each image with OpenCV.
3. Resizes images to `128 x 128` pixels.
4. Normalizes pixel values to the range `0.0` to `1.0`.
5. Groups training samples by person name.
6. Trains the CNN.
7. Saves the model and class mapping.

Generated files:

```text
model/face_model.h5
model/face_labels.json
```

Retrain the model whenever new people or new images are registered.

## Recognize a Person

1. Visit `http://127.0.0.1:5000/recognize`.
2. Upload an image.
3. Select **Recognize**.

The prediction route preprocesses the uploaded image in the same way as training, selects the highest-probability class, loads the corresponding name from `face_labels.json`, and displays the matching MongoDB record.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Registration page |
| `POST` | `/register` | Save a person and images |
| `GET` | `/recognize` | Image recognition page |
| `POST` | `/predict` | Predict the person in an uploaded image |

## Project Structure

```text
MachineLearning/
├── app.py                    # Flask application and routes
├── config.py                 # MongoDB settings and image size
├── train_model.py            # Loads MongoDB images and trains the CNN
├── pro3.ipynb                # Notebook used during experimentation
├── model/
│   ├── face_model.h5         # Generated trained model
│   └── face_labels.json      # Generated model-index-to-name mapping
├── templates/
│   ├── register.html         # Registration form
│   ├── recognise.html        # Recognition upload form
│   └── result.html           # Prediction result page
└── utils/
		├── cnn_model.py          # CNN architecture
		├── database.py           # MongoDB connection
		└── preprocess.py         # Image decoding and normalization
```

## Troubleshooting

### `No usable images found in MongoDB`

Register at least one person with a valid image before running `train_model.py`. Confirm that MongoDB is running and that `config.py` points to the correct database.

### `Model not trained yet`

Run `python train_model.py` first. Both `model/face_model.h5` and `model/face_labels.json` must exist.

### Predictions are inaccurate

The model is a small learning-project CNN and its accuracy depends heavily on the training data. Add several clear images per person, avoid mixing people in one image, keep faces visible, and retrain after adding data. A small dataset can memorize training images without generalizing well to new photos.

### TensorFlow cannot find CUDA

The application can run on the CPU. The CUDA warning only means TensorFlow is not using a compatible GPU.

### MongoDB connection errors

Start the MongoDB service and verify `MONGO_URI`, `DB_NAME`, and `COLLECTION_NAME` in `config.py`.

## Limitations

- This is an image-classification demo, not a production-grade face-recognition system.
- Images are stored directly in MongoDB, which may not be ideal for large datasets.
- There is no authentication or authorization.
- The application currently loads the model for each prediction request.
- Unknown-person detection and confidence thresholds are not implemented.
- Accuracy cannot be guaranteed with a small or unbalanced dataset.

## Possible Improvements

- Add data augmentation and a larger, balanced dataset.
- Use transfer learning with a pretrained vision model.
- Add prediction confidence and an unknown-person threshold.
- Cache the loaded model instead of loading it per request.
- Validate image size, file type, and upload limits.
- Add authentication and secure production configuration.
- Use a production WSGI server instead of Flask's development server.
