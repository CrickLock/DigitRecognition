import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cap=cv2.VideoCapture(0)
while True :
    try:
        ret, frame=cap.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width=gray.shape
        upperleft=(int(width/2 -56), int(height/2 -56))
        bottomright=(int(width/2 +56), int(height/2+56))
        cv2.rectangle(gray, upperleft, bottomright, (0, 255, 0), 2)
        roi=gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]
        Imagepil=Image.fromarray(roi)
        image_bw=Imagepil.convert("L")
        image_bw_resized=image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
        pixelfilter=20
        minpixel=np.percentile(image_bw_resized_inverted, pixelfilter)
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-minpixel, 0, 255)
        maxpixel=np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxpixel
        testsample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        testprediction=clf.predict(testsample)
        print(testprediction)
        cv2.imshow("frame", gray)
        

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()