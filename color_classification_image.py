  

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import sys

# read the test image
try:
    source_image = cv2.imread(sys.argv[1])
except:
    source_image = cv2.imread('image\/black_cat.jpg')
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
prediction = knn_classifier.main('training.data', 'test.data')
print('Detected color is:', prediction)
cv2.rectangle(source_image,(20,20), (300,60), (0,0,0), -1)
cv2.putText(
    source_image,
    'Prediction: ' + prediction,
    (50,50),
    2,
    0.8,
    (255,255,255),
    2,
    cv2.LINE_AA
    )

# Display the resulting frame
cv2.imshow('color classifier', source_image)
cv2.waitKey(0)		
