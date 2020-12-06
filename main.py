import tensorflow as tf
import plate_recognition_video
from cv2 import dnn_superres

yolo_model = tf.saved_model.load('./yolo_model') # model import

video_path = './videos/grupaA3.mp4'
video_out_path = './videos/grupaA3_detected.mp4'
txt_out_path = './videos/grupaA3_detected.txt'

sr_model = dnn_superres.DnnSuperResImpl_create()

#Setting path for model
#This loads all the variables of the chosen model and prepares the neural network for inference. The parameter is the path to your downloaded pre-trained model. 
sr_model_path = "FSRCNN_x4.pb"
sr_model.readModel(sr_model_path)

#Setting SR model
#1st par.: Name of the model which has to be correctly choosen based on model specified in sr.readModel()
#2nd par.: parameter Upscaling factor, i.e. how many times you will increase the resolution. Again, this needs to match with your chosen model

sr_model.setModel("fsrcnn", 4)

#tesseract_path = r'<ENTER YOUR PATH TO>\Tesseract-OCR\tesseract.exe'
tesseract_path = r'C:\Users\Mylosz\AppData\Local\Tesseract-OCR\tesseract.exe'


plate_recognition_video.plate_recognition(video_path, video_out_path, txt_out_path, yolo_model, sr_model, tesseract_path)