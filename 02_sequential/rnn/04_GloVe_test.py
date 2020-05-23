import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import numpy as np
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

imdb_dir = '/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/_dataSets/aclImdb'

maxlen = 100
training_samples = 200

validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)

test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                 labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
embedding_dim = 100

model = Sequential()

model.add(Embedding(max_words, embedding_dim,
                    input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

model.load_weights('models/pre_trained_glove_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)

print('test loss: ', test_loss, 'test acc: ', test_acc)