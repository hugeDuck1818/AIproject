import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask ,flash, request
from PIL import Image
import tensorflow as tf
import numpy as np

ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

model  = tf.keras.models.load_model("./deneme.h5")
app = Flask(__name__)

emos = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict",methods=["POST"])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
           
            return "No file uploaded" ,400
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
          
            return "No file uploaded" ,400
        if file and allowed_file(file.filename):
            image  = Image.open(file).convert('L').resize((48,48), Image.NEAREST)
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = input_arr/255
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            return emos[np.argmax(model.predict(input_arr))]
