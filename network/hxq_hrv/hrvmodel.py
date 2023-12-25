import functools

import numpy as np

import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model


print(tf.__version__)

hrv_data_path = "hrv_samples.csv"

LABEL_COLUMN = "result"
LABELS = [0, 1]

COLUMNS_TO_USE = [
    "sdnn",
    "sdann",
    "sdnn_index",
    "rmssd",
    "pnn50",
    "triangular_index",
    "hf",
    "lf",
    "result",
]


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=32,  # 为了示例更容易展示，手动设置较小的值
        label_name=LABEL_COLUMN,
        select_columns=COLUMNS_TO_USE,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
    )
    return dataset


hrv_data = get_dataset(hrv_data_path)


print(f"hrv_data: {hrv_data}")

examples, labels = next(iter(hrv_data))

print(f"examples: {examples}, labels: {labels}")


class HRVModel(Model):
    def __init__(self):
        super(HRVModel, self).__init__()
        self.d0 = Dense(8, activation="relu")
        self.d1 = Dense(8, activation="relu")
        self.d2 = Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.d0(x)
        x = self.d1(x)
        x = self.d1(x)
        return self.d2(x)


# Create an instance of the model
model = HRVModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()


model.compile(optimizer=optimizer, loss=loss_object, metrics=["accuracy"])


# model.fit(examples, labels, epochs=10)


def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    return tf.reshape(data, [-1, 1])


CATEGORIES = {}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab
    )
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))


MEANS = {
    "age": 29.631308,
    "n_siblings_spouses": 0.545455,
    "parch": 0.379585,
    "fare": 34.385399,
}

numerical_columns = []

for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(
        feature,
        normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]),
    )
    numerical_columns.append(num_col)

numerical_columns

preprocessing_layer = tf.keras.layers.DenseFeatures(
    categorical_columns + numerical_columns
)


model = keras.Sequential(
    [
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

train_data = hrv_data.shuffle(500)

model.fit(train_data, epochs=20)
