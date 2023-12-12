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
        png_files = sorted(png_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        word = ""
        for png_file in png_files:
            word += ocr.findLetter(png_file).lower()
        
        score = {}
        with open('mots.txt', 'r') as file:
            lines = file.readlines()
        mots = []

        for line in lines:
            mot = line.strip()
            mots.append(mot)

        for i, mot in enumerate(mots):
            score[i] = Levenshtein.compareString(word, mot)

        motTrouver = mots[min(score, key=score.get)]
        return [word, motTrouver]

    def processOne(self, file):
        ocr = Ocr()
        return ocr.findLetter(file)

