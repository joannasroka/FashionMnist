import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import save_model, load_model
 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
train_images = train_images / 255.0
test_images = test_images / 255.0
 
def get_random_eraser(p=0.15, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
 
        if p_1 > p:
            return input_img
 
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
 
            if left + w <= img_w and top + h <= img_h:
                break
 
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
 
        input_img[top:top + h, left:left + w, :] = c
 
        return input_img
 
    return eraser
 
#show first train image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
 
model = keras.Sequential()
# Conv + Maxpooling
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), padding='same', activation='relu', input_shape=(28,28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# Dropout
model.add(tf.keras.layers.Dropout(0.1))
# Conv + Maxpooling
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# Dropout
model.add(tf.keras.layers.Dropout(0.3))
# Flatting 3D feature to 1D feature vector
model.add(tf.keras.layers.Flatten(input_shape=(28,28, 1)))
# Fully connected Layer
model.add(tf.keras.layers.Dense(256, activation='relu'))
# Dropout
model.add(tf.keras.layers.Dropout(0.5))
# Fully Connected Layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))
 
model.summary()
 
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
             optimizer='adam',
             metrics=['accuracy'])
 
#print(train_images.shape)
 
#train_images = np.expand_dims(train_images, axis=-1)
#train_images = train_images.reshape(list(train_images.shape) + [1])
 
train_images = train_images.reshape((-1,28,28,1))
test_images = test_images.reshape((-1,28,28,1))
 
eraser = get_random_eraser()
 
for m in range(60000):
  eraser(train_images[m])
 
#print(train_images.shape)
#print(train_labels.shape)
 
model.fit(train_images, train_labels, epochs=50)
 
#model.save('my_model.h5')
 
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
 
print('\nTest accuracy:', test_acc)
 
 
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
 
predictions = probability_model.predict(test_images)
 
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
 
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
 
  plt.imshow(img, cmap=plt.cm.binary)
 
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
 
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
 
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
 
test_images_to_plot = test_images[:, :, :, 0]
print(f"Test images shape: {test_images.shape}")
 
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images_to_plot)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
 
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images_to_plot)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
 

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images_to_plot)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
