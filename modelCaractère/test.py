from PIL import Image
import json
from tensorflow import keras
import numpy as np
import glob
import os

# Load the trained model
loaded_model = keras.models.load_model('emnist_model.keras')

# Convert the prediction to a character label
character_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
png_files = glob.glob(os.path.join('car/', '*.jpg'))
n_car = 0
n_correct = 0
for image_path in png_files:
    input_image = Image.open(image_path)
    input_image = input_image.convert('L')  # Convert to grayscale
    input_image = input_image.resize((28, 28))  # Resize to the model's input size
    input_image = np.array(input_image)  # Convert to a NumPy array

    # Normalize the pixel values
    input_image = input_image.astype('float32') / 255.0

    # Reshape the input image to have the shape (1, 28, 28, 1)
    input_image = input_image.reshape(1, 28, 28, 1)
    image = image_path.split('.')
    image = image[0]
    image = image.split('/')
    image = image[1]
    # Make a prediction using the model
    prediction = loaded_model.predict(input_image)
    n_car += 1
    if character_labels[np.argmax(prediction)] == image[0].upper():
        n_correct +=1
    print(np.argmax(prediction))
    print(f'''vrai caractère : {image};   result : {character_labels[np.argmax(prediction)]}''')
print(f'Pourcentage de caractères trouvés: {n_correct / n_car * 100:.2f}%')