import numpy as np
import cv2
import os
import tensorflow as tf
from time import time
import plate_to_txt

def plate_recognition(video_path, video_out_path, txt_out_path, model):
    # Output file init
    txt_file = open(txt_out_path, 'w')

    # Video load and resize
    FRAME_SIZE = 416

    video_original = cv2.VideoCapture(video_path) # Video read
    
    video_width = int(video_original.get(3))
    video_height = int(video_original.get(4))
    video_fps = int(video_original.get(cv2.CAP_PROP_FPS))
    video_length = int(video_original.get(cv2.CAP_PROP_FRAME_COUNT)) # in frames

    video_output = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    time_start = time()

    skip = False # changes to true when there was no plates detected

    while (True): # breaks when frame is unavailable    
        
        if skip == True:
            for i in range(10):
                ret, frame_original = video_original.read()
                if ret == True:
                    video_output.write(frame_original)
                else:
                    break
            skip = False

        ret, frame_original = video_original.read()
        if ret == True: # if frame is available: ret = True
            # Changing format and size of a frame
            frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE,FRAME_SIZE))

            frame = frame / 255 # Frame normalization

            frames = []
            frames.append(frame)
            frames = np.array(frames).astype(np.float32)

            # Plate detecion
            data = tf.constant(frames)
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

            frame_result = frame_original.copy()
            plates_txt = []

            detected_plates_number = int(np.count_nonzero(boxes.numpy()[0]) / 4) # devided by 4 -> each plate has 4 coords
            for i in range(detected_plates_number):
                result = boxes.numpy()[0][i]

                # x_min, x_max, y_min, y_max -> coordinates of detected plate 
                x_min = int(result[1] * video_width)
                y_min = int(result[0] * video_height)
                x_max = int(result[3] * video_width)
                y_max = int(result[2] * video_height)
                
                if x_max - x_min > 90: # adding only plates wider than 90 px
                    x_min_cropped = int(x_min + (x_max - x_min)*0.1) # Cropping 10% of a plate from the left side
                                                             # so it doesnt try to recognize the character from the flag and country

                    cropped_plate = frame_original[y_min:y_max, x_min_cropped:x_max]
                    text = plate_to_txt.plate_txt(cropped_plate)
                    plates_txt.append(text)

                    frame_result = cv2.rectangle(frame_result,(x_min, y_min),(x_max, y_max),(0, 255, 0), 1) # Cropping plate from orginal frame
                    frame_result = cv2.putText(frame_result, text,(x_min, y_min), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1) # Cropping plate from orginal frame  

            if len(plates_txt) == 0:
                skip = True
            
            current_frame = int(video_original.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame == 1:
                # When it's the first frame of the video
                write_txt_log(txt_out_path, plates_txt, current_video_time(current_frame, video_fps), first = True)
            else:
                write_txt_log(txt_out_path, plates_txt, current_video_time(current_frame, video_fps))

            video_output.write(frame_result)

        else:
            break

    write_txt_log(txt_out_path, [], current_video_time(int(video_original.get(cv2.CAP_PROP_POS_FRAMES)), video_fps), last = True)

    avg_fps = video_length / (time() - time_start)
    print("Avg fps: " + str(avg_fps))

    video_output.release()

def write_txt_log(path, plates_txt, time, first = False, last = False):
    if first:
        # First detected frame -> create new txt file
        with open(path, 'w') as f:
            f.write("Liczba wykrytych tablic: " + str(len(plates_txt)) + "; " + ', '.join(str(plate) for plate in plates_txt) + "; czas: " + time + " - ")
    elif last:
        # Last frame
        with open(path, 'a') as f:
            f.write(time)
    else:
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]

            different = False # by default plates in previous frame and current frame are the same

            plates_last_line = last_line.split(';')[1].split(',')

            if len(plates_last_line) != len(plates_txt):
                # Different ammount of detected plates
                if len(plates_txt) != 0 or plates_last_line[0] != ' ': # Solved problem with no tables detected
                    different = True
            else:   
                for plate_txt in plates_txt:
                    if plate_txt not in last_line:
                        # Different plate detected
                        different = True

            if different == True:
                with open(path, 'a') as f:
                    f.write(time + "\n") # current time
                    f.write("Liczba wykrytych tablic: " + str(len(plates_txt)) + "; " + ', '.join(str(plate) for plate in plates_txt) + "; czas: " + time + " - ")

def current_video_time(current_frame, video_fps):
    seconds = int(current_frame / video_fps)

    minutes = int(seconds / 60)
    seconds %= 60
    
    return "%02d:%02d" % (minutes, seconds)
