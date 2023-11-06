import json
from service.levenshtein import Levenshtein
from service.ocr import Ocr
import numpy as np
import cv2
import os
import tempfile
import glob

class ProcessingPicture:

    def process(self, file):
        ocr = Ocr()
        output_dir = ocr.extractShapes(file)
        png_files = glob.glob(os.path.join(output_dir + '/', '*.png'))
        word = ""
        for png_file in png_files:
            word += ocr.findLetter(png_file)
        score = {}

        # for i, mot in enumerate(mots):
        #     score[i] = Levenshtein.compareString(content, mot)

        # motTrouver = mots[min(score, key=score.get)]
        return word

    def processOne(self, file):
        ocr = Ocr()
        return ocr.findLetter(file)

