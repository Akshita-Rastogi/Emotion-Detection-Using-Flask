from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf


def camera():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    global graph
    graph = tf.get_default_graph()
    with graph.as_default():
            classifier = load_model('Emotion_model.h5',compile=False)


    class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read() 
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 6)
        for (x,y,w,h) in faces:
        
            # Creating Rectangle around Face with color Red and of width 2
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Getting start and end points of face to detect eyes within the Face
            roi_gray = gray[y:y+h, x:x+w]  # for GrayScale
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
            
            
                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                cv2.putText(gray,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
            else:
                cv2.putText(frame,'No Face Found',cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
            cv2.imshow('Emotion Detector',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()