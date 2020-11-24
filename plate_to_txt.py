import cv2
import imutils
import numpy as np
import pytesseract

def plate_txt(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Mylosz\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

    #img = cv2.resize(img, (100,22) )

    #img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    #img = cv2.bilateralFilter(img,9,75,75)

    text = pytesseract.image_to_string(img, config='--psm 8 -c tessedit_char_whitelist=0123456789QWERTYUIOPASDFGHJKLZXCVBNM-')

    return text