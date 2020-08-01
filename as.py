# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:57:01 2020

@author: ASUS
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:22:10 2020

@author: ASUS
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential, load_model
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation
from keras.optimizers import Adam



human_angry = glob.glob("human_anger/*")

print("Number of images in Angry emotion = "+str(len(human_angry)))
human_angry_folderName = [str(i.split("\\")[0])+"/" for i in human_angry]
human_angry_imageName = [str(i.split("\\")[1]) for i in human_angry]
human_angry_emotion = [["Angry"]*len(human_angry)][0]
human_angry_label = [1]*len(human_angry)

len(human_angry_folderName), len(human_angry_imageName), len(human_angry_emotion), len(human_angry_label)
df_angry = pd.DataFrame()
df_angry["folderName"] = human_angry_folderName
df_angry["imageName"] = human_angry_imageName
df_angry["Emotion"] = human_angry_emotion
df_angry["Labels"] = human_angry_label
print(df_angry.head())



human_disgust = glob.glob("human_disgust/*")
print("Number of images in Disgust emotion = "+str(len(human_disgust)))

human_disgust_folderName = [str(i.split("\\")[0])+"/" for i in human_disgust]
human_disgust_imageName = [str(i.split("\\")[1]) for i in human_disgust]
human_disgust_emotion = [["Disgust"]*len(human_disgust)][0]
human_disgust_label = [2]*len(human_disgust)

len(human_disgust_folderName), len(human_disgust_imageName), len(human_disgust_emotion), len(human_disgust_label)

df_disgust = pd.DataFrame()
df_disgust["folderName"] = human_disgust_folderName
df_disgust["imageName"] = human_disgust_imageName
df_disgust["Emotion"] = human_disgust_emotion
df_disgust["Labels"] = human_disgust_label
print(df_disgust.head())


human_fear = glob.glob("human_fear/*")

print("Number of images in Fear emotion = "+str(len(human_fear)))

human_fear_folderName = [str(i.split("\\")[0])+"/" for i in human_fear]
human_fear_imageName = [str(i.split("\\")[1]) for i in human_fear]
human_fear_emotion = [["Fear"]*len(human_fear)][0]
human_fear_label = [3]*len(human_fear)

len(human_fear_folderName), len(human_fear_imageName), len(human_fear_emotion), len(human_fear_label)

df_fear = pd.DataFrame()
df_fear["folderName"] = human_fear_folderName
df_fear["imageName"] = human_fear_imageName
df_fear["Emotion"] = human_fear_emotion
df_fear["Labels"] = human_fear_label
print(df_fear.head())


human_happy = glob.glob("human_happy/*")

print("Number of images in  Happy emotion = "+str(len(human_happy)))
human_happy_folderName = [str(i.split("\\")[0])+"/" for i in human_happy]
human_happy_imageName = [str(i.split("\\")[1]) for i in human_happy]
human_happy_emotion = [["Happy"]*len(human_happy)][0]
human_happy_label = [4]*len(human_happy)

len(human_happy_folderName), len(human_happy_imageName), len(human_happy_emotion), len(human_happy_label)


df_happy = pd.DataFrame()
df_happy["folderName"] = human_happy_folderName
df_happy["imageName"] = human_happy_imageName
df_happy["Emotion"] = human_happy_emotion
df_happy["Labels"] = human_happy_label
print(df_happy.head())



human_neutral = glob.glob("human_neutral/*")

print("Number of images in Neutral emotion = "+str(len(human_neutral)))

human_neutral_folderName = [str(i.split("\\")[0])+"/" for i in human_neutral]
human_neutral_imageName = [str(i.split("\\")[1]) for i in human_neutral]
human_neutral_emotion = [["Neutral"]*len(human_neutral)][0]
human_neutral_label = [5]*len(human_neutral)

len(human_neutral_folderName), len(human_neutral_imageName), len(human_neutral_emotion), len(human_neutral_label)

df_neutral = pd.DataFrame()
df_neutral["folderName"] = human_neutral_folderName
df_neutral["imageName"] = human_neutral_imageName
df_neutral["Emotion"] = human_neutral_emotion
df_neutral["Labels"] = human_neutral_label
print(df_neutral.head())



human_sad = glob.glob("human_sad/*")

print("Number of images in Sad emotion = "+str(len(human_sad)))

human_sad_folderName = [str(i.split("\\")[0])+"/" for i in human_sad]
human_sad_imageName = [str(i.split("\\")[1]) for i in human_sad]
human_sad_emotion = [["Sad"]*len(human_sad)][0]
human_sad_label = [6]*len(human_sad)

len(human_sad_folderName), len(human_sad_imageName), len(human_sad_emotion), len(human_sad_label)

df_sad = pd.DataFrame()
df_sad["folderName"] = human_sad_folderName
df_sad["imageName"] = human_sad_imageName
df_sad["Emotion"] = human_sad_emotion
df_sad["Labels"] = human_sad_label
print(df_sad.head())


human_surprise = glob.glob("human_surprise/*")

print("Number of images in Surprise emotion = "+str(len(human_surprise)))

human_surprise_folderName = [str(i.split("\\")[0])+"/" for i in human_surprise]
human_surprise_imageName = [str(i.split("\\")[1]) for i in human_surprise]
human_surprise_emotion = [["Surprise"]*len(human_surprise)][0]
human_surprise_label = [7]*len(human_surprise)

len(human_surprise_folderName), len(human_surprise_imageName), len(human_surprise_emotion), len(human_surprise_label)
df_surprise = pd.DataFrame()
df_surprise["folderName"] = human_surprise_folderName
df_surprise["imageName"] = human_surprise_imageName
df_surprise["Emotion"] = human_surprise_emotion
df_surprise["Labels"] = human_surprise_label
print(df_surprise.head())


length = df_angry.shape[0] + df_disgust.shape[0] + df_fear.shape[0] + df_happy.shape[0] + df_neutral.shape[0] + df_sad.shape[0] + df_surprise.shape[0]
print("Total number of images in all the emotions = "+str(length))



frames = [df_angry, df_disgust, df_fear, df_happy, df_neutral, df_sad, df_surprise]
Final_human = pd.concat(frames)
print(Final_human.shape)

Final_human.reset_index(inplace = True, drop = True)
Final_human = Final_human.sample(frac = 1.0)   #shuffling the dataframe
Final_human.reset_index(inplace = True, drop = True)
print(Final_human.head())


df_human_train_data, df_human_test = train_test_split(Final_human, stratify=Final_human["Labels"], test_size = 0.215860)
print(df_human_train_data.shape)
print(df_human_test.shape)




df_human_train_data.reset_index(inplace = True, drop = True)
df_human_train_data.to_pickle("df_human_train.pkl")



df_human_test.reset_index(inplace = True, drop = True)
df_human_test.to_pickle("df_human_test.pkl")

df_human_train_data = pd.read_pickle("df_human_train.pkl")
print(df_human_train_data.head())

print(df_human_train_data.shape)


df_human_test = pd.read_pickle("df_human_test.pkl")
print(df_human_test.head())
print(df_human_test.shape)



df_temp_train = df_human_train_data.sort_values(by = "Labels", inplace = False)

df_temp_test = df_human_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_human_train_data["Emotion"].value_counts().sort_index()

TestData_distribution = df_human_test["Emotion"].value_counts().sort_index()

TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)

TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)






fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.2, y = i.get_height()+1.5, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)



fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of testing data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")





def convt_to_gray(df):
    count = 0
    for i in range(len(df)):
        path1 = df["folderName"][i]
        path2 = df["imageName"][i]
        img = cv2.imread(os.path.join(path1, path2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path1, path2), gray)
        count += 1
    print("Total number of images converted and saved = "+str(count))

"""
convt_to_gray(df_human_train)
convt_to_gray(df_human_test)





#detect the face in image using HAAR cascade then crop it then resize it and finally save it.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
#download this xml file from link: https://github.com/opencv/opencv/tree/master/data/haarcascades.
def face_det_crop_resize(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]  #cropping the face in image
        cv2.imwrite(img_path, cv2.resize(face_clip, (216, 216)))  #resizing image then saving it

for i, d in df_human_train_data.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)



for i, d in df_human_test.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)
"""
anime_angry = glob.glob("animated_anger/*")
print("Number of images in Angry emotion = "+str(len(anime_angry)))
anime_angry_folderName = [str(i.split("\\")[0])+"/" for i in anime_angry]
anime_angry_imageName = [str(i.split("\\")[1]) for i in anime_angry]
anime_angry_emotion = [["Angry"]*len(anime_angry)][0]
anime_angry_label = [1]*len(anime_angry)

len(anime_angry_folderName), len(anime_angry_imageName), len(anime_angry_emotion), len(anime_angry_label)
df_angry = pd.DataFrame()
df_angry["folderName"] = anime_angry_folderName
df_angry["imageName"] = anime_angry_imageName
df_angry["Emotion"] = anime_angry_emotion
df_angry["Labels"] = anime_angry_label
print(df_angry.head())





anime_disgust = glob.glob("animated_disgust/*")
print("Number of images in Disgust emotion = "+str(len(anime_disgust)))

anime_disgust_folderName = [str(i.split("\\")[0])+"/" for i in anime_disgust]
anime_disgust_imageName = [str(i.split("\\")[1]) for i in anime_disgust]
anime_disgust_emotion = [["Disgust"]*len(anime_disgust)][0]
anime_disgust_label = [2]*len(anime_disgust)

len(anime_disgust_folderName), len(anime_disgust_imageName), len(anime_disgust_emotion), len(anime_disgust_label)

df_disgust = pd.DataFrame()
df_disgust["folderName"] = anime_disgust_folderName
df_disgust["imageName"] = anime_disgust_imageName
df_disgust["Emotion"] = anime_disgust_emotion
df_disgust["Labels"] = anime_disgust_label
print(df_disgust.head())





anime_fear = glob.glob("animated_fear/*")
print("Number of images in Fear emotion = "+str(len(anime_fear)))
anime_fear_folderName = [str(i.split("\\")[0])+"/" for i in anime_fear]
anime_fear_imageName = [str(i.split("\\")[1]) for i in anime_fear]
anime_fear_emotion = [["Fear"]*len(anime_fear)][0]
anime_fear_label = [3]*len(anime_fear)

len(anime_fear_folderName), len(anime_fear_imageName), len(anime_fear_emotion), len(anime_fear_label)

df_fear = pd.DataFrame()
df_fear["folderName"] = anime_fear_folderName
df_fear["imageName"] = anime_fear_imageName
df_fear["Emotion"] = anime_fear_emotion
df_fear["Labels"] = anime_fear_label
print(df_fear.head())


anime_happy = glob.glob("animated_joy/*")
print("Number of images in Happy emotion = "+str(len(anime_happy)))
anime_happy_folderName = [str(i.split("\\")[0])+"/" for i in anime_happy]
anime_happy_imageName = [str(i.split("\\")[1]) for i in anime_happy]
anime_happy_emotion = [["Happy"]*len(anime_happy)][0]
anime_happy_label = [4]*len(anime_happy)

len(anime_happy_folderName), len(anime_happy_imageName), len(anime_happy_emotion), len(anime_happy_label)

df_happy = pd.DataFrame()
df_happy["folderName"] = anime_happy_folderName
df_happy["imageName"] = anime_happy_imageName
df_happy["Emotion"] = anime_happy_emotion
df_happy["Labels"] = anime_happy_label
print(df_happy.head())


anime_neutral = glob.glob("animated_neutral/*")
print("Number of images in Neutral emotion = "+str(len(anime_neutral)))

anime_neutral_folderName = [str(i.split("\\")[0])+"/" for i in anime_neutral]
anime_neutral_imageName = [str(i.split("\\")[1]) for i in anime_neutral]
anime_neutral_emotion = [["Neutral"]*len(anime_neutral)][0]
anime_neutral_label = [5]*len(anime_neutral)

len(anime_neutral_folderName), len(anime_neutral_imageName), len(anime_neutral_emotion), len(anime_neutral_label)

df_neutral = pd.DataFrame()
df_neutral["folderName"] = anime_neutral_folderName
df_neutral["imageName"] = anime_neutral_imageName
df_neutral["Emotion"] = anime_neutral_emotion
df_neutral["Labels"] = anime_neutral_label
print(df_neutral.head())



anime_sad = glob.glob("animated_sadness/*")
print("Number of images in Sad emotion = "+str(len(anime_sad)))

anime_sad_folderName = [str(i.split("\\")[0])+"/" for i in anime_sad]
anime_sad_imageName = [str(i.split("\\")[1]) for i in anime_sad]
anime_sad_emotion = [["Sad"]*len(anime_sad)][0]
anime_sad_label = [6]*len(anime_sad)

len(anime_sad_folderName), len(anime_sad_imageName), len(anime_sad_emotion), len(anime_sad_label)

df_sad = pd.DataFrame()
df_sad["folderName"] = anime_sad_folderName
df_sad["imageName"] = anime_sad_imageName
df_sad["Emotion"] = anime_sad_emotion
df_sad["Labels"] = anime_sad_label
print(df_sad.head())




anime_surprise = glob.glob("animated_surprise/*")
print("Number of images in Surprise emotion = "+str(len(anime_surprise)))

anime_surprise_folderName = [str(i.split("\\")[0])+"/" for i in anime_surprise]
anime_surprise_imageName = [str(i.split("\\")[1]) for i in anime_surprise]
anime_surprise_emotion = [["Surprise"]*len(anime_surprise)][0]
anime_surprise_label = [7]*len(anime_surprise)

len(anime_surprise_folderName), len(anime_surprise_imageName), len(anime_surprise_emotion), len(anime_surprise_label)

df_surprise = pd.DataFrame()
df_surprise["folderName"] = anime_surprise_folderName
df_surprise["imageName"] = anime_surprise_imageName
df_surprise["Emotion"] = anime_surprise_emotion
df_surprise["Labels"] = anime_surprise_label
print(df_surprise.head())


frames = [df_angry, df_disgust, df_fear, df_happy, df_neutral, df_sad, df_surprise]
Final_Animated = pd.concat(frames)
print(Final_Animated.shape)



df_anime_train_data, df_anime_test = train_test_split(Final_Animated, stratify=Final_Animated["Labels"], test_size = 0.198868)
print(df_anime_train_data.shape)
print( df_anime_test.shape)

df_anime_train_data.reset_index(inplace = True, drop = True)
df_anime_train_data.to_pickle("df_anime_train.pkl")

df_anime_test.reset_index(inplace = True, drop = True)
df_anime_test.to_pickle("df_anime_test.pkl")


df_anime_train_data = pd.read_pickle("df_anime_train.pkl")
print(df_anime_train_data.head())

print(df_anime_train_data.shape)


df_anime_test = pd.read_pickle("df_anime_test.pkl")
print(df_anime_test.shape)
print(df_anime_test.head())


df_temp_train = df_anime_train_data.sort_values(by = "Labels", inplace = False)
df_temp_test = df_anime_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_anime_train_data["Emotion"].value_counts().sort_index()
TestData_distribution = df_anime_test["Emotion"].value_counts().sort_index()
TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)
TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.185, y = i.get_height()+1.6, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)


fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.21, y = i.get_height()+0.3, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of test data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")
"""
print(convt_to_gray(df_anime_train_data))
print(convt_to_gray(df_anime_test))

"""

def change_image(df):
    count = 0
    for i, d in df.iterrows():
        img = cv2.imread(os.path.join(d["folderName"], d["imageName"]))
        face_clip = img[40:240, 35:225]         #cropping the face in image
        face_resized = cv2.resize(face_clip, (224, 224))
        cv2.imwrite(os.path.join(d["folderName"], d["imageName"]), face_resized) #resizing and saving the image
        count += 1
    print("Total number of images cropped and resized = {}".format(count))
"""
change_image(df_anime_train_data)
change_image(df_anime_test)
"""



frames = [df_human_train_data, df_anime_train_data]
combined_train = pd.concat(frames)
print(combined_train.shape)

combined_train = combined_train.sample(frac = 1.0)  #shuffling the dataframe
combined_train.reset_index(inplace = True, drop = True)
combined_train.to_pickle("combined_train.pkl")



Train_Combined = pd.read_pickle("combined_train.pkl")
num_classes=7

img_rows,img_cols = 48,48
batch_size = 32



train_data_dir='D:\\data science\\traindata'
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    shear_range=0.3,
                    zoom_range=0.3,
                    width_shift_range=0.4,
                    height_shift_range=0.4,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    color_mode='grayscale',
                    target_size=(img_rows,img_cols),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle='true')
test_data_dir='D:\\data science\\testdata'
test_batch = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_data_dir,
                                                      color_mode='grayscale',
                                                      target_size=(img_rows,img_cols),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle='false')
SAVER='D:\\data science'
                 
nb_classes = 7

# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau


checkpoint = ModelCheckpoint('D:\\data science\\Emotion_model.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)


earlystop = EarlyStopping( monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                          factor=0.2,
                          patience=3,
                          verbose=1,
                          min_delta=0.0001)
                    
callbacks=[earlystop,checkpoint,reduce_lr]


nb_train_samples=41929
epochs =25
history =model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            epochs=epochs,
            callbacks=callbacks)



model.save(os.path.join(SAVER, "Emotion_model.h5"))
#num_of_test_samples=5000
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




test_loss, test_acc = model.evaluate_generator(test_batch, steps=50)

print('test acc:', test_acc)
print('test loss:', test_loss)
num_of_test_samples = 12253
test_steps_per_epoch = np.math.ceil(test_batch.samples / test_batch.batch_size)

predictions = model.predict_generator(test_batch, steps=test_steps_per_epoch)
#predictions = model.predict_generator(test_batch,  num_of_test_samples // batch_size+1)

y_pred = np.argmax(predictions, axis=1)

true_classes = test_batch.classes

class_labels = list(test_batch.class_indices.keys())   

print(class_labels)

print(confusion_matrix(test_batch.classes, y_pred))

report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)


print(Train_Combined.shape)
