import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

# example based on GloVe
# Unter https://nlp.stanford.edu/projects/glove
# können Sie die vorab be- rechnete Einbettung der englischen
# Wikipedia 2014 herunterladen. Die 882 MB große Zip-Datei heißt glove.6B.zip
# und enthält 100-dimensionale Einbettungs- vektoren für 400.000 Wörter
# (oder Tokens, die keine Wörter sind)

#/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/_dataSets/aclImdb

#/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/_models/glove.6B


imdb_dir = '/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/_dataSets/aclImdb'

train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


maxlen = 100
training_samples = 200

validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('%s valid tokens found.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

print('Shape data tensor:', data.shape)
print('Shape classification tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples +
                               validation_samples]
y_val = labels[training_samples: training_samples +
                                 validation_samples]

glove_dir = '/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/_models/glove.6B'

embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('%s word vectors found.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()

model.add(Embedding(max_words, embedding_dim,
                    input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('models/pre_trained_glove_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(0)
plt.plot(epochs, loss, 'bo', label='Loss Train')
plt.plot(epochs, val_loss, 'b', label='Loss Val')
plt.title('Loss Train/Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(epochs, acc, 'bo', label='Train Acc')
plt.plot(epochs, val_acc, 'b', label='Val Acc')
plt.title('Rate Train/Val')
plt.xlabel('Epochs')
plt.ylabel('Rate')
plt.legend()
plt.show()
