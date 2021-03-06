import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask ,flash, request,jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
from lyrics import posSong , negSong
from flask_cors import CORS, cross_origin
from random import randint

negEmo = ["angry","disgust","fear","sad"]
posEmo = ["happy","surprise"]

posSongLen = len(posSong)-1
negSongLen = len(negSong)-1
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

model  = tf.keras.models.load_model("./deneme.h5")
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

emos = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def randomVerses(song):
    lyrics = song["lyrics"]
    splitted = lyrics.splitlines()
    lenSplitted = len(splitted)
    idx = randint(0,lenSplitted-2)
    while(splitted[idx]=="" or splitted[idx][0]=="["):
        idx =  randint(0,lenSplitted-2)
    return [splitted[idx] , splitted[idx+1]]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=["GET"])
@cross_origin()
def root():
    return "<h1>Here is the app<h1>"
    
@app.route("/predict",methods=["POST"])
@cross_origin()
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
            emotion = emos[np.argmax(model.predict(input_arr))]
            if(emotion in posEmo):
                song = posSong[randint(0,posSongLen)]
                
            else:
                song = negSong[randint(0,negSongLen)]

            song["lyrics"] = randomVerses(song)
            return jsonify({"emotion": emotion,"song":song})
