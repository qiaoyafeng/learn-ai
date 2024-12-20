import numpy as np
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.preprocessing import sequence

from tcn import TCN

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential([
    Embedding(max_features, 128, input_shape=(maxlen,)),
    TCN(kernel_size=6, dilations=[1, 2, 4, 8, 16]),
    Dense(1, activation='sigmoid')
])

print(f'TCN receptive field: {model.layers[1].receptive_field}.')

model.summary()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    validation_data=[x_test, y_test]
)

model_name = 'tcn_imdb.h5'

model.save(model_name)

my_tcn = tensorflow.keras.models.load_model(model_name, custom_objects={'TCN': TCN})


# Make inference.
preds = my_tcn.predict(x_test)
print('*' * 80)
print('Inference :', preds)

for ind, val in enumerate(preds):
    print(f'id : {ind}, pred: {val}, y_test: {y_test[ind]}')


