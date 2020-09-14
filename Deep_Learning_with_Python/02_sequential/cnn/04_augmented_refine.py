import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

# _______________________________________________________________________________________________________
# importing pre-trained model

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.summary()

# _______________________________________________________________________________________________________
# defining data-paths

#base_dir = '/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/dogs-vs-cats/cats_and_dogs_small'
base_dir = 'D:\Dropbox\work\_SquareHouse\_CodeBase\dogs-vs-cats\cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# _______________________________________________________________________________________________________
# randomizing data via generator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory( train_dir,
#                                                      target_size=(150, 150),
#                                                      batch_size=20,
#                                                      class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory( validation_dir,
#                                                         target_size=(150, 150),
#                                                         batch_size=20,
#                                                         class_mode='binary')
#
# # _______________________________________________________________________________________________________
# # adding new/untrained dense layers to the pre-trained "head" / convolutional layers
#
# model = models.Sequential()
#
# model.add(conv_base)
#
# model.add(layers.Flatten())
#
# model.add(layers.Dropout(0.5))
#
# model.add(layers.Dense(256, activation='relu'))
#
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.summary()
#
# # _______________________________________________________________________________________________________
# # setting last 3 conv-layers back TO trainable
#
# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
# # _______________________________________________________________________________________________________
# # compile and fit model
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-5),
#               metrics=['acc'])
#
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=100,
#       validation_data=validation_generator,
#       validation_steps=50)
#
# # _______________________________________________________________________________________________________
# # data output
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
#
# def smooth_curve(points, factor=0.8):
#     smoothed_points = []
#     for point in points:
#       if smoothed_points:
#           previous = smoothed_points[-1]
#           smoothed_points.append(previous * factor +
#                                  point * (1 - factor))
#       else:
#           smoothed_points.append(point)
#     return smoothed_points
#
# plt.figure(0)
# plt.plot(epochs, smooth_curve(loss), 'bo', label='Loss Train')
# plt.plot(epochs, smooth_curve(val_loss), 'b', label='Loss Val')
# plt.title('Loss Train/Val')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.figure(1)
# plt.plot(epochs, smooth_curve(acc), 'bo', label='Train Acc')
# plt.plot(epochs, smooth_curve(val_acc), 'b', label='Val Acc')
# plt.title('Rate Train/Val')
# plt.xlabel('Epochs')
# plt.ylabel('Rate')
# plt.legend()
# plt.show()
#
# model.save('models/cats_and_dogs_small_3.h5')
#
# # del model


model = models.load_model('models/cats_and_dogs_small_3.h5')

test_generator = test_datagen.flow_from_directory( test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_loss, test_acc =  model.evaluate_generator(test_generator, steps=50)

print('Corr Rate Test:', test_acc)