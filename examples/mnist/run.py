# pylint: disable=E1101,R,C
import os
import sys
sys.path.append("../../")

import numpy as np
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import tensorflow as tf
import gzip
import pickle
import numpy as np
import argparse
from tqdm import tqdm

MNIST_PATH = "s2_mnist.gz"

DEVICE = "/gpu:0" if tf.config.experimental.list_physical_devices("GPU") else "/cpu:0" 

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-3


def save_model(path, model):
    if not os.path.exists(path):
        print('save directories...', flush=True)
        os.makedirs(path)
    model.save_weights(path + '/smnist_model')


def load_data(path, batch_size):
    
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = tf.convert_to_tensor(dataset["train"]["images"][:, None, :, :].astype(np.float64))
    train_labels = tf.convert_to_tensor(dataset["train"]["labels"].astype(np.int64))

    test_data = tf.convert_to_tensor(dataset["test"]["images"][:, None, :, :].astype(np.float64))
    test_labels = tf.convert_to_tensor(dataset["test"]["labels"].astype(np.int64))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

    return train_dataset, test_dataset


class S2ConvNet_original(tf.keras.Model):
    def __init__(self):
        super(S2ConvNet_original, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 30
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

        self.fc = tf.keras.layers.Dense(f_output)

    def call(self, x):
        x = tf.reshape(x, (x.shape[0],1,60,60))
        x = self.conv1(x)
        x = tf.keras.activations.relu(x)

        x = self.conv2(x)
        x = tf.keras.activations.relu(x)

        x = so3_integrate(x)

        x = self.fc(x)
        return x


class S2ConvNet_deep(tf.keras.Model):

    def __init__(self, bandwidth=30):
        super(S2ConvNet_deep, self).__init__()

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = tf.keras.Sequential([
            S2Convolution(
                nfeature_in  = 1,
                nfeature_out = 8,
                b_in  = bandwidth,
                b_out = bandwidth,
                grid=grid_s2),
            tf.keras.layers.ReLU(),

            SO3Convolution(
                nfeature_in  =  8,
                nfeature_out = 16,
                b_in  = bandwidth,
                b_out = bandwidth//2,
                grid=grid_so3_1),
            tf.keras.layers.ReLU(),

            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 16,
                b_in  = bandwidth//2,
                b_out = bandwidth//2,
                grid=grid_so3_2),
            tf.keras.layers.ReLU(),

            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 24,
                b_in  = bandwidth//2,
                b_out = bandwidth//4,
                grid=grid_so3_2),
            tf.keras.layers.ReLU(),

            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 24,
                b_in  = bandwidth//4,
                b_out = bandwidth//4,
                grid=grid_so3_3),
            tf.keras.layers.ReLU(),

            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 32,
                b_in  = bandwidth//4,
                b_out = bandwidth//8,
                grid=grid_so3_3),
            tf.keras.layers.ReLU(),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = bandwidth//8,
                b_out = bandwidth//8,
                grid=grid_so3_4),
            tf.keras.layers.ReLU()
        ])

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10)
        ])

    def call(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def main(network):
    NUM_EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.lr)

    train_dataset, test_dataset = load_data(MNIST_PATH, BATCH_SIZE)

    if network == 'original':
        classifier = S2ConvNet_original()
    elif network == 'deep':
        classifier = S2ConvNet_deep()
    else:
        raise ValueError('Unknown network architecture')

    # print("#params", sum(np.prod(var.shape) for var in classifier.trainable_variables))

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    classifier.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # history = classifier.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset, verbose=2)

    for epoch in range(NUM_EPOCHS):
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset)):
            with tf.GradientTape() as tape:
                logits = classifier(x_batch_train, training=True)
                loss_value = criterion(y_batch_train, logits)
            grads = tape.gradient(loss_value, classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads, classifier.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in test_dataset:
            val_logits = classifier(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))

    save_model('models', classifier)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",
                        help="network architecture to use",
                        default='original',
                        choices=['original', 'deep'])
    parser.add_argument("--epochs",
                        help="number of epochs to run for",
                        default=20)
    parser.add_argument("--lr",
                        help="learning rate",
                        default=5e-3)
    args = parser.parse_args()

    main(args.network)
