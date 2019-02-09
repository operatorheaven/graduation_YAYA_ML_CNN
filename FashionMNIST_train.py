import tensorflow as tf 
import keras
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,\
        Convolution2D,MaxPooling2D

nb_filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)

fashion_mnist = keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) =\
    fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0
train_images, test_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1),\
    test_images.reshape(test_images.shape[0], 28, 28, 1)


#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
    Convolution2D(nb_filters, 
                  (kernel_size[0],kernel_size[1]),
                    padding='same',
                    input_shape=(28, 28, 1)),
    Activation('relu'),
    Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
