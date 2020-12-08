import numpy as np
import cv2
import os
import tensorflow as tf
from time import time
from plate_to_txt import plate_txt

def plate_recognition(video_path, video_out_path, txt_out_path, yolo_model, sr_model, tesseract_path):
    # Output file init
    txt_file = open(txt_out_path, 'w')

    FRAME_SIZE = 416
    video_original = cv2.VideoCapture(video_path) # Video read
    
    # Geting info about the video
    video_width = int(video_original.get(3))
    video_height = int(video_original.get(4))
    video_fps = int(video_original.get(cv2.CAP_PROP_FPS))
    video_length = int(video_original.get(cv2.CAP_PROP_FRAME_COUNT)) # in frames

    # Initialization of the result video
    video_output = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    # Start to measure time of the recognition process
    time_start = time()

    while (True): # breaks when it fails to read frame   
        
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
            infer = yolo_model.signatures['serving_default']
            
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

            plates_txt = []
            plates_coords = []

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
                                                                     # so it doesnt attempt to recognize the character from the flag and country
                    cropped_plate = frame_original[y_min:y_max, x_min_cropped:x_max] # cropped plate from original frame
                    text = plate_txt(cropped_plate, sr_model, tesseract_path)
                    plates_txt.append(text)
                    plates_coords.append(((x_min, y_min),(x_max, y_max)))

            current_frame = int(video_original.get(cv2.CAP_PROP_POS_FRAMES))
            video_time = current_video_time(current_frame, video_fps)
            if current_frame == 1:
                # When it's the first frame of the video
                write_txt_log(txt_out_path, plates_txt, video_time, first = True)
            else:
                write_txt_log(txt_out_path, plates_txt, video_time)

            if len(plates_txt) == 0: # When there is no plates detected
                video_output.write(frame_original)

                for i in range(10): # skips 10 frames if there is no plates detected
                    ret, frame_original = video_original.read()
                    if ret == True:
                        video_output.write(frame_original)
                    else:
                        break

            else: # When there is at least one plate detected
                # Mark the same plates in current and next plate (saves time)
                frames = [frame_original]
                ret, frame_original_2 = video_original.read()
                if ret == True:
                    frames.append(frame_original_2)
                
                for frame in frames: 
                    frame_result = frame.copy()

                    for i in range(len(plates_txt)):
                        frame_result = cv2.rectangle(frame_result, plates_coords[i][0], plates_coords[i][1], (0, 255, 0), 1)
                        frame_result = cv2.putText(frame_result, plates_txt[i], plates_coords[i][0], cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                    
                    video_output.write(frame_result)
        else:
            break # breaks from loop when frame is unavailable (video ends)

    write_txt_log(txt_out_path, [], current_video_time(int(video_original.get(cv2.CAP_PROP_POS_FRAMES)), video_fps), last = True)

    avg_fps = video_length / (time() - time_start)

    video_output.release()

    return avg_fps

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
                if len(plates_txt) != 0 or plates_last_line[0] != ' ': # Solved problem (when no tables were detected)
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
    # Returning video time from the given frame number
    seconds = int(current_frame / video_fps)

    minutes = int(seconds / 60)
    seconds %= 60
    
    return "%02d:%02d" % (minutes, seconds)
