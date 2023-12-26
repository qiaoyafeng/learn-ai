import functools
import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers


print(tf.__version__)

hrv_data_path = "hrv_samples_test.csv"

hrv_train = pd.read_csv(
    hrv_data_path,
    names=[
        "index",
        "sdnn",
        "sdann",
        "sdnn_index",
        "rmssd",
        "pnn50",
        "triangular_index",
        "hf",
        "lf",
        "result",
    ],
)

print(f"hrv_train head: {hrv_train.head()}")

hrv_features = hrv_train.copy()
hrv_labels = hrv_features.pop("result")
hrv_indexes = hrv_features.pop("index")

hrv_test = hrv_train.sample(50)

print(f"hrv_test: {hrv_test}")

hrv_test_features = hrv_test.copy()
hrv_test_labels = hrv_test_features.pop("result")
hrv_test_indexes = hrv_test_features.pop("index")


hrv_test_features = np.array(hrv_test_features)

hrv_features = np.array(hrv_features)
print(f"hrv_features: {hrv_features}")

##  模型A
# hrv_model = tf.keras.Sequential(
#     [
#         layers.Dense(64, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(1),
#     ]
# )
#
# hrv_model.compile(
#     loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam()
# )
#
# hrv_model.fit(hrv_features, hrv_labels, epochs=50)

##


normalize = layers.Normalization()

normalize.adapt(hrv_features)


norm_hrv_model = tf.keras.Sequential(
    [normalize, layers.Dense(64, activation="relu"), layers.Dense(1)]
)

norm_hrv_model.compile(
    loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy']
)

norm_model_history = norm_hrv_model.fit(
    hrv_features,
    hrv_labels,
    epochs=500,
    verbose=1,
    validation_data=(hrv_test_features, hrv_test_labels),
)

results = norm_hrv_model.evaluate(hrv_test_features, hrv_test_labels, verbose=2)

print(f"norm_hrv_model evaluate result: {results}")


history_dict = norm_model_history.history
print(f"history_dict: {history_dict}")

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, label="Training loss")
# b代表“蓝色实线”
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


plt.clf()   # 清除数字

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

