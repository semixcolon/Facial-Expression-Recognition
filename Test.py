from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import keras
from keras.utils.generic_utils import CustomObjectScope

#with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#classifier = load_model('mobilenet.h5') #for mobilenet
#classifier = load_model('inceptionv3.h5') #for inceptionv3
classifier = load_model('FER',compile=False)

face_classifier = cv2.CascadeClassifier('cascade.xml')

#for video
#cap = cv2.VideoCapture('test.mp4') 
#for webcam
cap = cv2.VideoCapture(0)
class_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral','Contempt']

from keras.utils.generic_utils import CustomObjectScope


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #for mobilenet and inceptionv3 model
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        #roi_gray = cv2.resize(roi_gray,(224,224),interpolation=cv2.INTER_AREA) #for mobilenet
        #roi_gray = cv2.resize(roi_gray,(299,299),interpolation=cv2.INTER_AREA) #for inceptionv3
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()
