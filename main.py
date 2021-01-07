import tensorflow as tf
from plate_recognition_video import plate_recognition
from video_validate import video_validate
from cv2 import dnn_superres
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QPushButton, QGridLayout, QInputDialog, QLineEdit, QFileDialog, QProgressBar
from PyQt5.QtGui import QIcon
import os

yolo_model = tf.saved_model.load('./yolo_model') # model import


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
        self.processVideoButton.setEnabled(False)
        self.playVideoButton = QPushButton("&Play Video", self)
        self.playVideoButton.setEnabled(False)
        self.viewLogButton = QPushButton("&View Log", self)
        self.viewLogButton.setEnabled(False)

        self.filePath = QLineEdit()
        self.filePath.setReadOnly(True)

        self.chooseFileButton.clicked.connect(self.openFileDialog)
        self.processVideoButton.clicked.connect(self.processVideo)
        self.playVideoButton.clicked.connect(self.playVideo)
        self.viewLogButton.clicked.connect(self.viewLog)

        hView = QHBoxLayout()
        hView.addWidget(self.chooseFileButton)
        hView.addWidget(self.processVideoButton)
        hView.addWidget(self.playVideoButton)
        hView.addWidget(self.viewLogButton)

        self.finishLabel = QLabel(self)
        self.finishLabel.setAlignment(QtCore.Qt.AlignCenter)

        tableView.addLayout(hView, 0, 0, 2, 2)
        tableView.addWidget(self.filePath, 1, 0, 2, 2)
        tableView.addWidget(self.finishLabel, 3,0,2,2)
        
        self.setLayout(tableView)

        self.setGeometry(20, 20, 500, 200)
        self.setWindowIcon(QIcon('plate.png'))
        self.setWindowTitle("Number Plate Recognition App")
        self.show()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose video file path...", "","Video Files (*.mp4 *.avi *.mov *.mkv)", options=options)
        if fileName:
            self.video_path=fileName
            self.video_out_path = fileName[0:-4] + "_detected" + fileName[-4:]
            self.txt_out_path = fileName[0:-4] + "_detected.txt"
            self.filePath.setText(self.video_path)
            self.playVideoButton.setEnabled(False)
            self.viewLogButton.setEnabled(False)
            self.processVideoButton.setEnabled(True)
            self.finishLabel.setText("")
            
    def processVideo(self):
        validation = video_validate(self.video_path)
        if validation == 0:
            self.filePath.setEnabled(False)
            self.chooseFileButton.setEnabled(False)
            avg_fps = plate_recognition(self.video_path, self.video_out_path, self.txt_out_path, yolo_model, sr_model, tesseract_path)
            self.finishLabel.setText("FINISHED !!!  AVG FPS: " + str(avg_fps))
            self.playVideoButton.setEnabled(True)
            self.viewLogButton.setEnabled(True)
            self.filePath.setEnabled(True)
            self.chooseFileButton.setEnabled(True)
        else:
            self.finishLabel.setText(str(validation))

    def playVideo(self):
        os.system("start " + self.video_out_path.replace(' ', '" "'))
    def viewLog(self):
        os.system("start " + self.txt_out_path.replace(' ', '" "'))



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    okno = MainWindow()
    sys.exit(app.exec_())


