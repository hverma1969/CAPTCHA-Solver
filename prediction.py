'''
This will be main file which the co-ordinaters of the event will be using to test your
code. This file contains two functions:

1. predict: You will be given an rgb image which you will use to predict the output 
which will be a string. For the prediction you can use/import code,models from other files or
libraries. More detailes given above the function defination.

2. test: This will be used by the co-ordinators to test your code by giving sample 
inputs to the 'predict' function mentioned above. A sample test function is given for your
reference but it is subject to minor changes during the evaluation. However, note that
there won't be any changes in the input format given to the predict function.

Make sure all the necessary functions etc. you import are done from the same directory. And in 
the final submission make sure you provide them also along with this script.
'''


import tensorflow as tf
# from tensorflow import keras
# from keras import layers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
# from tesseract_predict import *
import os
import glob
import shutil
import sys
import numpy as np
import cv2
from PIL import Image
import imutils
from preprocessing import preprocess 
# from preprocessing1 import preprocess
from keras.models import load_model

'''
function: predict
input: image - A numpy array which is an rgb image
output: answer - A string which is the full captcha

Suggestion: Try to make your code understandable by including comments and organizing it. For 
this we encourgae you to write essential function in other files and import them here so that 
the final code is neat and not too big. Make sure you use the same input format and return 
same output format.
'''
def predict(image):
    model=models.Sequential()
    num_classes=26
    model.add(layers.Conv2D(32,(5,5),padding='valid',activation='relu',input_shape=(64,64,1)))
    model.add(layers.Conv2D(64,(5,5),padding='valid',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.Conv2D(256,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes,activation='softmax'))
    model.load_weights('model_char5.h5')
    model2 = load_model('46_model.h5')
    class_mapping='0LACdEhHJKMn2PRTUWX3Y5bQ89'
    class_mapping2='0123?56?89A8C?EF?H?JKLM?0PQR5TU?WXY2Qbd???hn9??'
    
    count=0
    total=0
    images,images2=preprocess(image)
    answer=""
    for i in range(len(images2)):
        image1=images[i]
        image2=images2[i]
        result = np.argmax(model.predict(image1))
        result_confidence=np.max(model.predict(image1))
        print("model1",class_mapping[result],result_confidence)
        result2 = np.argmax(model2.predict(image2))
        result_confidence2=np.max(model2.predict(image2))
        print("model2",class_mapping2[result2],result_confidence2)
        if class_mapping2[result2]=='?':
            answer+=(class_mapping[result])
        elif class_mapping[result]=='Q':
            answer+=class_mapping2[result2]
        else:
            if result_confidence2<=result_confidence-0.14:
                answer+=(class_mapping[result])
            else:
                answer+=(class_mapping2[result2])
    return answer            
'''
    Write your code for prediction here.
    '''
    # answer = 'xyzabc' # sample needs to be modified



'''
function: test
input: None
output: None

This is a sample test function which the co-ordinaors will use to test your code. This is
subject to change but the imput to predict function and the output expected from the predict
function will not change. 
You can use this to test your code before submission: Some details are given below:
image_paths : A list that will store the paths of all the images that will be tested.
correct_answers: A list that holds the correct answers
score : holds the total score. Keep in mind that scoring is subject to change during testing.

You can play with these variables and test before final submission.
'''
def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = ['corthon.jpeg']
    correct_answers = ['AXCKP']
    score = 0

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a string is expected
        print(answer)
        if correct_answers[i] == answer:
            score += 10
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()