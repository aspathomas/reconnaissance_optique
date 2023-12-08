import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization

# Load the image data
image_path = 'lettres.png'
img = Image.open(image_path).convert('L')
img = img.resize((28, 28))
img_array = np.array(img)

# Normalize and reshape the input data
img_array = img_array.astype('float32') / 255.0
img_array = img_array.reshape(1, 28, 28, 1)
nb_classes = 62
labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
label_indices = {char: i for i, char in enumerate(labels)}
labels = [label_indices[char] for char in labels]
print(labels)
print(nb_classes)
labels = np.array(labels).reshape(1, nb_classes)



model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(nb_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Compile and train the model
model.fit(img_array, labels, epochs=10, verbose=1)

# Evaluate the model
score = model.evaluate(img_array, labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save the model to a file
model.save('emnist_model.keras')