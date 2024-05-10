import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# class_names = ['Apple Apple scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)healthy', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
class_names = [
    'Apple Apple scab', 
    'Apple Black rot', 
    'Apple Cedar apple rust', 
    'Apple healthy', 
    # 'Blueberry healthy', 
    'Cherry (including sour) Powdery mildew', 
    'Cherry (including sour) healthy', 
    'Corn (maize) Cercospora leaf spot Gray leaf spot', 
    'Corn (maize) Common rust', 
    'Corn (maize) Northern Leaf Blight', 
    'Corn (maize) healthy', 
    'Grape Black rot', 
    'Grape Esca (Black Measles)', 
    'Grape Leaf blight (Isariopsis Leaf Spot)', 
    'Grape healthy', 
    # 'Orange Haunglongbing (Citrus greening)', 
    'Peach Bacterial spot', 
    'Peach healthy', 
    'Pepper, bell Bacterial spot', 
    'Pepper, bell healthy', 
    'Potato Early blight', 
    'Potato Late blight', 
    'Potato healthy', 
    # 'Raspberry healthy', 
    # 'Soybean healthy', 
    # 'Squash Powdery mildew', 
    'Strawberry Leaf scorch', 
    'Strawberry healthy', 
    'Tomato Bacterial spot', 
    'Tomato Early blight', 
    'Tomato Late blight', 
    'Tomato Leaf Mold', 
    'Tomato Septoria leaf spot', 
    'Tomato Spider mites Two-spotted spider mite', 
    'Tomato Target Spot', 
    'Tomato Tomato Yellow Leaf Curl Virus', 
    'Tomato Tomato mosaic virus', 
    'Tomato healthy'
]


model_path = "trained_modelff.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError("Model file not found at {}".format(model_path))

def predict_disease(image_path):
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return class_names[result_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disease-info')
def disease_info():
    return render_template('disease_info.html')


@app.route('/upload', methods=['GET', 'POST'])  # Change the path to '/upload'
def upload_file():
    if request.method == 'POST':
        file = request.files.get('my_image')
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the file
            prediction = predict_disease(file_path)
            # Redirect using the URL path parameters
            return redirect(url_for('result', prediction=prediction, filename=filename))
    return render_template('upload.html')

@app.route('/result/<prediction>/<filename>')
def result(prediction, filename):
    img_path = url_for('static', filename='uploads/' + filename)
    return render_template('result.html', prediction=prediction, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)