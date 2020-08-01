#Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import  load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import os
import cv2
from camera import camera
# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.get_default_graph()

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = load_img(filename, color_mode="grayscale", target_size=(48, 48))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel

    img = img.reshape(1, 48, 48, 1)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
    
        filename = file.filename
        file_path = os.path.join('static/images', filename)
        file.save(file_path)
        img = read_image(file_path)
                # Predict the class of an image
        print()
        print()
        print(img)
        print()
        print()
        with graph.as_default():
            model1 = load_model('Emotion_model.h5',compile=False)
            print()
            print()
            print(img)
            print()
            print()
            class_prediction = np.argmax(model1.predict(img))
            class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
            print(class_labels[class_prediction])

                #Map apparel category with the numerical class
                
        return render_template('predict.html', product = class_labels[class_prediction], user_image = file_path)

    return render_template('predict.html')
@app.route("/display")
def display():
    init()
    camera()



def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


    
if __name__ == "__main__":
    init()
    app.run(debug=True)