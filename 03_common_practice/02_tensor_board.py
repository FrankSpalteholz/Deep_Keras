import tensorflow as tf
import datetime
import tensorflow.keras as keras
import keras.backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Embedding, Dropout, Input, Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence


max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()

model.add(keras.layers.Embedding(max_features, 128,
                           input_length=max_len,
                           name='embed'))

model.add(keras.layers.Conv1D(32, 7, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(32, 7, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(1))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# callbacks = [keras.callbacks.TensorBoard(
#             log_dir='/Users/frankfurt/PycharmProjects/Deep/keras_learning_01/tb_log',
#             histogram_freq=1,
#             embeddings_freq=1,
#             )
# ]

log_dir = "tb_log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1,)

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=[tensorboard_callback])


# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# run: tensorboard --logdir /Users/frankfurt/PycharmProjects/Deep/keras_learning_01/tb_log ... in terminal
