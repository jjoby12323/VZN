import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import matplotlib.pyplot as plt

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

width = 800
height = 600

img = cv2.imread("test images/tftestpic1.jpg")
img = cv2.resize(img, (width, height))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rgb_tensor = tf.convert_to_tensor(rgb, dtype = tf.uint8)
rgb_tensor = tf.expand_dims(rgb_tensor, 0)

label = pd.read_csv("labels.csv", sep=';', index_col='ID')
labels = label['OBJECT (2017 REL.)']

boxes, scores, classes, num_detection = detector(rgb_tensor)
num_detection

pred_labels = classes.numpy().astype('int')[0]
pred_labels = [labels[i] for i in pred_labels]

pred_boxes = boxes.numpy()[0].astype('int')
pred_scores = scores.numpy()[0]

for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < 0.5:
        continue
    img_boxes = cv2.rectangle(rgb, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_boxes, label, (xmin, ymin-10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    score_txt = f"{100 * round(score)}%"
    cv2.putText(img_boxes, score_txt, (xmin+100, ymin-10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
plt.imshow(img_boxes)
plt.show()
