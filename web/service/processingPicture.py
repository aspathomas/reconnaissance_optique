from PIL import Image
import pytesseract
import json
from service.levenshtein import Levenshtein

class ProcessingPicture:

    def process(self, file):
        img = Image.open(file)
        content = pytesseract.image_to_string(img)
        score = {}
        with open('mots.txt', 'r') as file_test:
            file_content = file_test.read()
        mots = file_content.split('\n')

        for i, mot in enumerate(mots):
            score[i] = Levenshtein.compareString(content, mot)

        motTrouver = mots[min(score, key=score.get)]
        return motTrouver

