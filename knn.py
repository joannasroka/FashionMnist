import tensorflow as tf
from tensorflow import keras
 
 
import matplotlib.pyplot as plt
import numpy as np
 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
train_images = (np.array(train_images) / 255.0).reshape(-1, 784).astype('float32')
train_labels = np.array(train_labels).astype('uint8')
test_images = (np.array(test_images) / 255.0).reshape(-1, 784).astype('float32')
test_labels = np.array(test_labels).astype('uint8')
 
classes = 10
 
 
def euclidean_distance(X, X_train): #obliczanie odleglosci euklidesowej
    return -2 * np.dot(X, X_train.T) + np.sum(X_train**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
 
 
 
def sort_labels_dist(Dist, y):  # sortowanie etykiet wzgledem odleglosci
    N1, N2 = Dist.shape
    temp = np.argsort(Dist)
    Dist = Dist.astype('uint8')
    for i in range(N1):
        Dist[i] = y[temp[i]]
    return Dist
 
 
def count_p_y_x(y, k):  # szukanie rozkladu prawdopodobienstwa p(y|x) dla kazdej z klas dla k najblizszych sasiadow
    y_k = np.delete(y, range(k, y.shape[1]), axis=1)
    occurrences = [np.bincount(y_k[i], minlength=classes) for i in range(y_k.shape[0])]
    occurrences = np.asarray(occurrences)
 
    return occurrences / k
 
 
def get_y(p_y_x):  # wyznaczanie etykiet na podstawie p(y|x)
    y = [np.apply_along_axis(np.argmax, arr=p_y_x, axis=1)]
    y = np.asarray(y)
    y = np.transpose(y)
 
    return y
 
 
def find_best_k(x_train, y_train, x_val, y_val, k_value):  # szukanie k, dla ktorego predykcja jest najdokladniejsza
    dist = euclidean_distance(x_val, x_train)
    sorted_labels = sort_labels_dist(dist, y_train)
    max_accuracy = 0
    max_accuracy_k = 1000
    for k in range(1, k_value):
        prob = count_p_y_x(sorted_labels, k)
        prediction = get_y(prob)
        accuracy = check_accuracy(prediction, y_val)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_k = k
 
    return max_accuracy
 
 
def check_accuracy(y_result, y_true):  # sprawdzenie dokladnosci predykcji
    equal = 0
    for i in range(y_result.shape[0]):
        if y_result[i] == y_true[i]:
            equal += 1
 
    return equal / y_result.shape[0]
 
print('\nTest accuracy:', find_best_k(train_images, train_labels, test_images, test_labels, 5))