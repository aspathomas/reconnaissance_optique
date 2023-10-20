from PIL import Image
import json
from service.levenshtein import Levenshtein
from tensorflow import keras
import numpy as np

class ProcessingPicture:

    # Load the trained model
    loaded_model = keras.models.load_model('emnist_model.keras')

    # Convert the prediction to a character label
    character_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    def process(self, file):

         # Load and preprocess the input image
        input_image = Image.open(file)
        input_image = input_image.convert('L')  # Convert to grayscale
        input_image = input_image.resize((28, 28))  # Resize to the model's input size
        input_image = np.array(input_image)  # Convert to a NumPy array
        input_image = input_image.reshape(1, 28 * 28)  # Flatten the image

        # Normalize the pixel values
        input_image = input_image.astype('float32') / 255.0

        # Make a prediction using the model
        prediction = self.loaded_model.predict(input_image)
        predicted_character = self.character_labels[np.argmax(prediction)]

        # score = {}
        # with open('mots.txt', 'r') as file_test:
        #     file_content = file_test.read()
        # mots = file_content.split('\n')

        # for i, mot in enumerate(mots):
        #     score[i] = Levenshtein.compareString(content, mot)

        # motTrouver = mots[min(score, key=score.get)]
        return predicted_character

