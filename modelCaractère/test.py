from PIL import Image
import json
from tensorflow import keras
import numpy as np
import glob
import os


loaded_model = keras.models.load_model('emnist_model.keras')

character_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
png_files = glob.glob(os.path.join('car/', '*.jpg'))
png_files += glob.glob(os.path.join('car2/', '*.png'))
png_files += glob.glob(os.path.join('car3/', '*.png'))
n_car = 0
n_correct = 0
for image_path in png_files:
    input_image = Image.open(image_path)
    input_image = input_image.convert('L')
    input_image = input_image.resize((28, 28))
    input_image = np.array(input_image)

    input_image = input_image.astype('float32') / 255.0

    input_image = input_image.reshape(1, 28, 28, 1)
    image = image_path.split('.')
    image = image[0]
    image = image.split('/')
    image = image[1]
    prediction = loaded_model.predict(input_image)
    n_car += 1
    if character_labels[np.argmax(prediction)] == image[0].upper():
        n_correct +=1
    print(np.argmax(prediction))
    print(f'''vrai caractère : {image};   result : {character_labels[np.argmax(prediction)]}''')
print(f'Pourcentage de caractères trouvés: {n_correct / n_car * 100:.2f}%')