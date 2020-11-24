import numpy as np
import cv2
import os
import tensorflow as tf
import plate_to_txt

'''Model import'''
path = os.path.abspath(os.getcwd())
model = tf.saved_model.load(path + '\yolo_model')

'''Image load and resize'''
IMAGE_SIZE = 416

img_original = cv2.imread(path + '/images/photo7.png') # Image read
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) # change img to black and white
img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))

img = img / 255 # Image normalization

images = []
images.append(img)

images = np.asarray(images).astype(np.float32)

'''Plate detecion'''
data = tf.constant(images)
infer = model.signatures['serving_default']
predictions = infer(data)

for key, value in predictions.items():
    boxes = value[:, :, 0:4]
    pred_conf = value[:, :, 4:]

# 'boxes' varialble contains coordinates of detected plates
boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=50,
    max_total_size=50,
    iou_threshold=0.45,
    score_threshold=0.5
)

'''Detected plates marking'''
img_result = img_original.copy()
original_h, original_w, _ = img_original.shape
plates = []

detected_plates_number = int(np.count_nonzero(boxes.numpy()[0]) / 4) # devided by 4 -> each plate has 4 coords
for i in range(detected_plates_number):
    result = boxes.numpy()[0][i]

    # x_min, x_max, y_min, y_max -> coordinates of detected plate 
    x_min = int(result[1] * original_w)
    y_min = int(result[0] * original_h)
    x_max = int(result[3] * original_w)
    y_max = int(result[2] * original_h)

    plates.append(img_original[y_min:y_max, x_min:x_max])

    img_result = cv2.rectangle(img_result,(x_min, y_min),(x_max, y_max),(0, 255, 0), 1)

'''Print plates text'''
for plate in plates:
    print("Wykryta tablica: " + str(plate_to_txt.plate_txt(plate)))

'''Showing the result image'''
cv2.imshow('img', img_result)

cv2.waitKey(0) # waiting until window is closed
cv2.destroyAllWindows() # end program
