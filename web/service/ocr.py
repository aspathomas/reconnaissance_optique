import numpy as np
from tensorflow import keras
from PIL import Image
import tempfile
import cv2
import os

class Ocr:
    loaded_model = keras.models.load_model('emnist_model.keras')
    character_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def extractShapes(self, file):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            input_image = Image.open(file)
            input_image.save(temp_file.name)
        source_image = cv2.imread(temp_file.name)
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(source_gray, 128, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda arr: arr[0][0][0])
        output_dir = tempfile.mkdtemp()

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            shape = source_gray[y:y+h, x:x+w]

            border_size = 5
            shape_with_border = cv2.copyMakeBorder(shape, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            shape_filename = os.path.join(output_dir, f"letter_{i}.png")
            cv2.imwrite(shape_filename, shape_with_border)

        return output_dir

    def findLetter(self, file):
        input_image = Image.open(file)
        input_image = input_image.convert('L')
        input_image = input_image.resize((28, 28))
        input_image = np.array(input_image)

        input_image = input_image.astype('float32') / 255.0

        input_image = input_image.reshape(1, 28, 28, 1)
        prediction = self.loaded_model.predict(input_image)
        return self.character_labels[np.argmax(prediction)]