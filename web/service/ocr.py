import numpy as np
from tensorflow import keras
from PIL import Image
import tempfile
import cv2
import os

class Ocr:
    # Load the trained model
    loaded_model = keras.models.load_model('emnist_model.keras')

    # Convert the prediction to a character label
    character_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def extractShapes(self, file):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            input_image = Image.open(file)
            input_image.save(temp_file.name)
        source_image = cv2.imread(temp_file.name)
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(source_gray, 128, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a temporary directory to save the extracted shapes
        output_dir = tempfile.mkdtemp()

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            shape = source_gray[y:y+h, x:x+w]
            shape_filename = os.path.join(output_dir, f"letter_{i}.png")
            cv2.imwrite(shape_filename, shape)

        return output_dir

    def findLetter(self, file):
        input_image = Image.open(file)
        input_image = input_image.convert('L')  # Convert to grayscale
        input_image = input_image.resize((28, 28))  # Resize to the model's input size
        input_image = np.array(input_image)  # Convert to a NumPy array

        # Normalize the pixel values
        input_image = input_image.astype('float32') / 255.0

        # Reshape the input image to have the shape (1, 28, 28, 1)
        input_image = input_image.reshape(1, 28, 28, 1)
        # Make a prediction using the model
        prediction = self.loaded_model.predict(input_image)
        return self.character_labels[np.argmax(prediction)]