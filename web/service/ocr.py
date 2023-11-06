import numpy as np
from tensorflow import keras
from PIL import Image

class Ocr:
    # Load the trained model
    loaded_model = keras.models.load_model('emnist_model.keras')

    # Convert the prediction to a character label
    character_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def findLetter(file):
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