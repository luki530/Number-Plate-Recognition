import tensorflow as tf
from plate_recognition_video import plate_recognition
from cv2 import dnn_superres
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QPushButton, QGridLayout, QInputDialog, QLineEdit, QFileDialog, QProgressBar
from PyQt5.QtGui import QIcon
import os

yolo_model = tf.saved_model.load('./yolo_model') # model import

video_path = ''
video_out_path = ''
txt_out_path = ''

sr_model = dnn_superres.DnnSuperResImpl_create()

#Setting path for model
#This loads all the variables of the chosen model and prepares the neural network for inference. The parameter is the path to your downloaded pre-trained model. 
sr_model_path = "FSRCNN_x4.pb"
sr_model.readModel(sr_model_path)

#Setting SR model
#1st par.: Name of the model which has to be correctly choosen based on model specified in sr.readModel()
#2nd par.: parameter Upscaling factor, i.e. how many times you will increase the resolution. Again, this needs to match with your chosen model

sr_model.setModel("fsrcnn", 4)

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.interface()

    def interface(self):

        tableView = QGridLayout()

        self.chooseFileButton = QPushButton("&Choose File", self)
        self.processVideoButton = QPushButton("&Process Video", self)
        self.playVideoButton = QPushButton("&Play Video", self)
        self.playVideoButton.setEnabled(False)
        self.viewLogButton = QPushButton("&View Log", self)
        self.viewLogButton.setEnabled(False)

        self.filePath = QLineEdit()
        self.filePath.setReadOnly(True)
        self.filePath.setText(video_path)

        self.chooseFileButton.clicked.connect(self.openFileDialog)
        self.processVideoButton.clicked.connect(self.processVideo)
        self.playVideoButton.clicked.connect(self.playVideo)
        self.viewLogButton.clicked.connect(self.viewLog)

        hView = QHBoxLayout()
        hView.addWidget(self.chooseFileButton)
        hView.addWidget(self.processVideoButton)
        hView.addWidget(self.playVideoButton)
        hView.addWidget(self.viewLogButton)

        self.progressBar = QProgressBar(self)

        finishLabel = QLabel(self)
        finishLabel.setAlignment(QtCore.Qt.AlignCenter)

        tableView.addLayout(hView, 0, 0, 2, 2)
        tableView.addWidget(self.filePath, 1, 0, 2, 2)
        tableView.addWidget(self.progressBar, 2, 0, 2, 2)
        tableView.addWidget(finishLabel, 3,0,2,2)

        #
        self.progressBar.setValue(50)
        
        self.setLayout(tableView)

        self.setGeometry(20, 20, 500, 200)
        self.setWindowIcon(QIcon('plate.png'))
        self.setWindowTitle("Number Plate Recognition App")
        self.show()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose video file path...", "","Video Files (*.mp4)", options=options)
        if fileName:
            video_path=fileName
            video_out_path = fileName[0:-4] + "_detected" + fileName[-4:]
            txt_out_path = fileName[0:-4] + "_detected.txt"
            self.filePath.setText(video_path)
            
    def processVideo(self):
        self.filePath.setEnabled(False)
        self.chooseFileButton.setEnabled(False)
        avg_fps = plate_recognition(video_path, video_out_path, txt_out_path, yolo_model, sr_model, tesseract_path)
        self.finishLabel.setText("FINISHED !!!  AVG FPS: " + avg_fps)
        self.playVideoButton.setEnabled(True)
        self.viewLogButton.setEnabled(True)

    def playVideo(self):
        os.system("start " + video_out_path)
    def viewLog(self):
        os.system("start " + txt_out_path)



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    okno = MainWindow()
    sys.exit(app.exec_())


