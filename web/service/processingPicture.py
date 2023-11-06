import json
from service.levenshtein import Levenshtein
from service.ocr import Ocr
import numpy as np
import cv2
import os
import tempfile

class ProcessingPicture:

    # Load the trained model
    loaded_model = keras.models.load_model('emnist_model.keras')

    # Convert the prediction to a character label
    character_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def extractShapes(file):
        source_image = cv2.imread(file)
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(source_gray, 128, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a temporary directory to save the extracted shapes
        output_dir = tempfile.mkdtemp()

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            shape = source_gray[y:y+h, x:x+w]
            shape_filename = os.path.join(output_dir, f"letter_{i}.jpg")
            cv2.imwrite(shape_filename, shape)

        return output_dir

    def process(self, file):
        output_dir = self.extractShapes(file)
        png_files = glob.glob(os.path.join(output_dir + '/', '*.png'))
        word = ""
        for png_file in png_files:
            word += Ocr.findLetter(png_file)
        # score = {}
        # with open('mots.txt', 'r') as file_test:
        #     file_content = file_test.read()
        # mots = file_content.split('\n')

        # for i, mot in enumerate(mots):
        #     score[i] = Levenshtein.compareString(content, mot)

        # motTrouver = mots[min(score, key=score.get)]
        return predicted_character

