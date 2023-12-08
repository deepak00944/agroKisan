import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
# from skimage import io
from tensorflow.keras.preprocessing import image
# from keras.models import Sequential
# from tensorflow._api.v1.keras import layers

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pandas as pd
import pickle

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__, static_folder=r"C:\Users\Deepak\Desktop\crop\static")

model2 = tf.keras.models.load_model("best_plant_model.h5")
data = pd.DataFrame(pd.read_csv("final.csv"))
model = pickle.load(open(r"model.pkl", "rb"))
crop = pd.read_csv(r"final_dataset.csv").copy()
area = pd.read_csv(r"final_data.csv")
commodity = pd.read_csv(r"commodities.csv")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictor')
def predictor():
    season = crop['Season'].unique()
    return render_template('predictor.html', seasons=season)


@app.route('/market')
def market():
    state1 = sorted(area['State'].unique())
    # print(states)  # Add this line to print the states

    area["District"] = area["State"] + "_" + area["District"]
    district1 = sorted(area['District'].unique())
    # print(district1)
    # area['Market'] = area['District'] + "_" + area['Market']
    markets = sorted(area['Market'])
    commodities = commodity["Commodities"].unique()
    return render_template('market.html', states=state1, districts=district1, commodities=commodities, markets=markets)


@app.route('/commodities')
def commodities():
    state1 = sorted(area['State'].unique())

    crop["District"] = area["State"] + "_" + area["District"]
    district1 = sorted(area['District'].unique())

    return render_template('commodities.html', states=state1, district1=district1)


@app.route('/predict', methods=['POST'])
def predict():
    pH = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))
    temperature = float(request.form.get('temperature'))
    nitrogen = int(request.form.get('nitrogen'))
    phosphorus = int(request.form.get('phosphorus'))
    potassium = int(request.form.get('potassium'))
    humidity = float(request.form.get('humidity'))
    output = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]]),

    return ("{} Crop can be Grown".format(str(output[0])))


def about_disease(filtered_df, column):
    cause_values = filtered_df[column]

    f = ""
    for cause in cause_values:
        f = f + cause + ", "  # Separate multiple causes with a comma
    return f.rstrip(", ")  # Remove trailing comma and spaces


def load_prep(img_path):
    img = tf.io.read_file(img_path)

    img = tf.image.decode_image(img)

    img = tf.image.resize(img, size=(224, 224))

    return img


def model_predict(img_path, model2):
    image = load_prep(img_path)
    preds = model2.predict(tf.expand_dims(image, axis=0))
    return preds


@app.route('/disease', methods=['GET'])
def disease():
    # Main page
    return render_template('disease.html')


@app.route('/disease_pred', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model2)
        print(preds)

        # x = x.reshape([64, 64]);
        disease_class = ['Apple___Apple_scab',
                         'Apple___Black_rot',
                         'Apple___Cedar_apple_rust',
                         'Apple___healthy',
                         'Blueberry___healthy',
                         'Cherry_(including_sour)___Powdery_mildew',
                         'Cherry_(including_sour)___healthy',
                         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                         'Corn_(maize)___Common_rust_',
                         'Corn_(maize)___Northern_Leaf_Blight',
                         'Corn_(maize)___healthy',
                         'Grape___Black_rot',
                         'Grape___Esca_(Black_Measles)',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                         'Grape___healthy',
                         'Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot',
                         'Peach___healthy',
                         'Pepper,_bell___Bacterial_spot',
                         'Pepper,_bell___healthy',
                         'Potato___Early_blight',
                         'Potato___Late_blight',
                         'Potato___healthy',
                         'Raspberry___healthy',
                         'Soybean___healthy',
                         'Squash___Powdery_mildew',
                         'Strawberry___Leaf_scorch',
                         'Strawberry___healthy',
                         'Tomato___Bacterial_spot',
                         'Tomato___Early_blight',
                         'Tomato___Late_blight',
                         'Tomato___Leaf_Mold',
                         'Tomato___Septoria_leaf_spot',
                         'Tomato___Spider_mites Two-spotted_spider_mite',
                         'Tomato___Target_Spot',
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                         'Tomato___Tomato_mosaic_virus',
                         'Tomato___healthy']

        result = disease_class[preds.argmax()]

        filtered_df = data[data['Type'] == result]

        symptoms = about_disease(filtered_df, 'Symptoms')
        cause = about_disease(filtered_df, 'Cause')
        prevention = about_disease(filtered_df, 'Prevention')

        response = {
            'disease': result,
            'cause': cause,
            'symptoms': symptoms,
            'prevention': prevention
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run();