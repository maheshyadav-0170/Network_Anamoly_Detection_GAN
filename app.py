
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template

app = Flask(__name__)

# Suppress TensorFlow info messages
tf.get_logger().setLevel('ERROR')

# Suppress stdout to hide progress updates


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open('null', 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout


# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file.
dataset = pd.read_csv('network_anomaly_dataset.csv')

# Preprocess the dataset
numeric_columns = dataset.select_dtypes(include=[np.number])
numeric_columns = numeric_columns.fillna(0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_columns)

# Split the dataset into training and test sets
train_data, test_data = train_test_split(
    scaled_data, test_size=0.2, random_state=42)

# Define the GAN model
latent_dim = 32

# Generator
generator = keras.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.2),
    layers.Dense(256),
    layers.LeakyReLU(alpha=0.2),
    layers.Dense(scaled_data.shape[1])
])

# Discriminator
discriminator = keras.Sequential([
    layers.Input(shape=(scaled_data.shape[1],)),
    layers.Dense(256),
    layers.LeakyReLU(alpha=0.2),
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.2),
    layers.Dense(1, activation='sigmoid')
])

discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training the GAN with 'train_data'
epochs = 1
batch_size = 32

# The main function to run your anomaly detection code


def run_anomaly_detection():
    for e in range(epochs):
        for _ in range(len(train_data) // batch_size):
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_data = generator.predict(noise)
            real_data = train_data[np.random.randint(
                0, train_data.shape[0], size=batch_size)]
            data = np.concatenate([real_data, generated_data])
            labels = np.zeros(2 * batch_size)
            labels[:batch_size] = 1
            discriminator.train_on_batch(data, labels)
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            labels = np.ones(batch_size)
            gan.train_on_batch(noise, labels)

# Anomalies detection with MSE using 'test_data'


def detect_anomalies():
    noise = np.random.normal(0, 1, size=[len(test_data), latent_dim])
    generated_data = generator.predict(noise)
    mse = np.mean(np.square(generated_data - test_data), axis=1)
    threshold = 0.6
    anomalies = np.where(mse > threshold)
    anomaly_indices = anomalies[0]
    non_anomaly_indices = np.delete(np.arange(len(test_data)), anomaly_indices)

    # Count the lengths
    total_test_data = len(test_data)
    num_detected_anomalies = len(anomaly_indices)
    num_non_anomalies = len(non_anomaly_indices)

    detected_anomalies = [dataset.iloc[index] for index in anomaly_indices]
    non_anomalies = [dataset.iloc[index] for index in non_anomaly_indices]

    return total_test_data, num_detected_anomalies, num_non_anomalies, detected_anomalies, non_anomalies


# Define routes for the web application


@app.route('/')
def index():
    return render_template('index.html')

# Define route for running anomaly detection

# Route for running anomaly detection


@app.route('/run_detection', methods=['POST'])
def run_detection():
    run_anomaly_detection()  # Run the anomaly detection code
    total_test_data, num_detected_anomalies, num_non_anomalies, detected_anomalies, non_anomalies = detect_anomalies()
    return render_template('result.html', total_test_data=total_test_data, num_detected_anomalies=num_detected_anomalies, num_non_anomalies=num_non_anomalies, detected_anomalies=detected_anomalies, non_anomalies=non_anomalies)

# Rest of your app.py code


if __name__ == '__main__':
    app.run(debug=True)
