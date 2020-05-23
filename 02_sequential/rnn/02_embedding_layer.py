from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras import preprocessing

max_features = 10000

maxlen = 20

# reading data into integer-lists
(x_train, y_train), (x_test, y_test) = imdb.load_data( num_words=max_features)

# 2d-tensor ... adding 10000 words from imdb add cut each entry after 20 words
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


model = Sequential()

# 8-dimensional embedding
# Legt die maximale Länge der Eingabe für den Einbettungslayer fest,
# damit er später in die eingebetteten Eingaben umgewandelt werden können.
# Nach dem Ein- bettungslayerbesitzendieAktivierungendieShape(samples, maxlen, 8).

model.add(Embedding(10000, 8, input_length=maxlen))

# Wandelt den 3-D-Tensor der Einbettungen in einen 2-D-Tensor mit der Shape (samples, maxlen * 8) um
model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
