from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.utils.np_utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

epochs = 10
batch_size = 512

deep_layer_dense = 16
output_layer_dense = 1
dropout_rate = 0.5

save_model_path = "/cnn/models/"
save_model_name = "model_v001.tf"
save_model_flag = 1;
load_model_flaf = 0;

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]



model = models.Sequential()

# model.add(layers.Dense(deep_layer_dense,
#                        kernel_regularizer = regularizers.l2(0.001),
#                        activation='relu',
#                        input_shape=(10000,)))

# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(deep_layer_dense,
#                        kernel_regularizer = regularizers.l2(0.001),
#                        activation='relu'))
#
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(output_layer_dense, activation='sigmoid'))
#
#
#
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_data=(x_val, y_val))
#
#
#
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values) + 1)
#
#
# #=============================================================================================
#
# plt.figure(0)
# plt.plot(epochs, loss_values, 'bo', label='Loss Train')
# plt.plot(epochs, val_loss_values, 'b', label='Loss Val')
# plt.title('Loss Train/Val')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.figure(1)
# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
# plt.plot(epochs, acc, 'bo', label='Train')
# plt.plot(epochs, val_acc, 'b', label='Val')
# plt.title('Rate Train/Val')
# plt.xlabel('Epochs')
# plt.ylabel('Rate')
# plt.legend()
# plt.show()
#
# #========================================================================================================
#
# if save_model_flag:
#     model.save(save_model_path + save_model_name,True,True)

del model


# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu',
#                            input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=4, batch_size=512)
# results = model.evaluate(x_test, y_test)

model = models.load_model(save_model_path+save_model_name)
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print(results)



