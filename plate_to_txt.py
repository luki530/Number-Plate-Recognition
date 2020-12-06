  
import cv2
import imutils
import numpy as np
import pytesseract
from cv2 import dnn_superres
from scipy.ndimage import interpolation as inter

def upscale_image(img):
    # Creating an Super Resolution(SR) object
    sr = dnn_superres.DnnSuperResImpl_create()

    #Setting path for model
    #This loads all the variables of the chosen model and prepares the neural network for inference. The parameter is the path to your downloaded pre-trained model. 
    path = "EDSR_x4.pb"
    sr.readModel(path)

    #Setting SR model
    #1st par.: Name of the model which has to be correctly choosen based on model specified in sr.readModel()
    #2nd par.: parameter Upscaling factor, i.e. how many times you will increase the resolution. Again, this needs to match with your chosen model
    sr.setModel("edsr", 4)

    #This is the inference part, which runs your image through the neural network and produces your upscaled image.
    img = sr.upsample(img)

    return img

def clean_image(img):
    # Denoising image with implementation of Non-local Means Denoising algorithm
    # Used to remove noise from color images. (Noise is expected to be gaussian).
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    # Filter enhances the details, and makes the image look sharper.
    img = cv2.detailEnhance(img, 10, 0.2)

    # Convolves an image with the kernel.
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    #It erodes away the boundaries of foreground object (Always try to keep foreground in white)
    # So the thickness or size of the foreground object decreases or simply white region decreases in the image. 
    # It is useful for removing small white noises (as we have seen in colorspace chapter), detach two connected objects etc.
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)

    return img


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def plate_txt(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    # Straightening the image
    angle,img = correct_skew(img)   

    # Upscaling the image
    img = upscale_image(img)

    # Cleaning the image
    img = clean_image(img)

    cv2.imshow('car',img)

    #reading the text from the image
    text = pytesseract.image_to_string(img, config=' --psm 8 -c tessedit_char_whitelist=0123456789QWERTYUIOPASDFGHJKLZXCVBNM')

    return text