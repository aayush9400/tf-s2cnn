# pylint: disable=E1101,R,C
import sys
sys.path.append("../../")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gzip
import pickle
import numpy as np

MNIST_PATH = "s2_mnist.gz"

DEVICE = "/gpu:0" if tf.config.experimental.list_physical_devices("GPU") else "/cpu:0"

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-4


class ConvNet(models.Model):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = layers.Conv2D(32, (5, 5), strides=(3,3), activation='relu')
        self.conv2 = layers.Conv2D(64, (5, 5), strides=(3,3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def load_data(path, batch_size):
    
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = tf.convert_to_tensor(dataset["train"]["images"][:, None, :, :].astype(np.float64))
    train_labels = tf.convert_to_tensor(dataset["train"]["labels"].astype(np.int64))

    test_data = tf.convert_to_tensor(dataset["test"]["images"][:, None, :, :].astype(np.float64))
    test_labels = tf.convert_to_tensor(dataset["test"]["labels"].astype(np.int64))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)
    # for i in train_dataset.as_numpy_iterator():
    #     print(i[0].shape, i[1].shape)
    #     break
    return train_dataset, test_dataset


def main():

    train_dataset, test_dataset = load_data(MNIST_PATH, BATCH_SIZE)

    classifier = ConvNet()

    print("#params", sum([x.numel() for x in classifier.parameters()]))


    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    classifier.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

    history = classifier.fit(train_dataset, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=test_dataset)

    return history, classifier


if __name__ == '__main__':
    history, model = main()
