import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train_en = to_categorical(Y_train, 10)
Y_test_en = to_categorical(Y_test, 10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, Y_train_en, epochs=20, verbose=1, validation_data=(X_test, Y_test_en))

evaluation_results = model.evaluate(X_test, Y_test_en)
print(f'Test Loss: {evaluation_results[0]}, Test Accuracy: {evaluation_results[1]}')



random_index = np.random.randint(0, len(X_test))

plt.imshow(X_test[random_index])
plt.show()

input_image = np.expand_dims(X_test[random_index], axis=0)
predictions = model.predict(input_image)

predicted_class_index = np.argmax(predictions)
predicted_class_name = class_names[predicted_class_index]
print(f"The image is a: {predicted_class_name}")
