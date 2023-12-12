import numpy as np
from keras.models import Sequential
from keras.src.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import os
import cv2

# Hypothetical dataset directory containing images of letters
dataset_dir = 'car'

# Load and preprocess the dataset
def load_and_preprocess_data(dataset_dir):
    data = []
    labels = []
    n = 5
    for n1 in range(n):
        for n2 in range(n):
            for n3 in range(n):
                for n4 in range(n):
                    for dataset_dir in ['car', 'car2', 'car3']:
                        for letter in os.listdir(dataset_dir):
                            letter_path = os.path.join(dataset_dir, letter)
                            # print(f'Label: {letter[0].upper()}, Image Path: {letter_path}')
                            img = load_img(letter_path, target_size=(28, 28), grayscale=True)
                            img_array = img_to_array(img)
                            img_with_border = cv2.copyMakeBorder(img_array, n1*10, n2*10, n3*10, n4*10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                            resized_img = cv2.resize(img_with_border, (28, 28))
                            data.append(resized_img)
                            labels.append(letter[0].upper())

    data = np.array(data, dtype='float32') / 255.0
    labels = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    print(len(categorical_labels))
    return data, categorical_labels


# Load and preprocess the data
data, labels = load_and_preprocess_data(dataset_dir)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=128, validation_data=(test_data, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
# Save the model to a file
model.save('emnist_model.keras')