import tensorflow as tf
import plate_recognition_video

model = tf.saved_model.load('./yolo_model') # model import

video_path = './videos/grupaA3.mp4'
video_out_path = './videos/grupaA3_detected.mp4'
txt_out_path = './videos/grupaA3_detected.txt'

plate_recognition_video.plate_recognition(video_path, video_out_path, txt_out_path, model)