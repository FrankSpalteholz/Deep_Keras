# ModelCheckPoint and EarlyStopping
# Early Stopping is used for stopping fit.function before over-fitting happens
# ModelCheckPoint is used for saving out versions of the models

import keras
callbacks_list = [ keras.callbacks.EarlyStopping(
                    monitor='acc',
                    patience=1,
                    ),
                    keras.callbacks.ModelCheckpoint(
                    filepath='my_model.h5',
                    monitor='val_loss',
                    save_best_only=True,) ]

model.compile(  optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))


# Callback ReduceLROnPlateau
# used for exiting local minima

callbacks_list = [  keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=10,) ]

model.fit(x, y,
            epochs=10,
            batch_size=32,
            callbacks=callbacks_list,
            validation_data=(x_val, y_val))

# Wenn Sie während des Trainings eine bestimmte Aktion ausführen möchten, die sich
# mit den integrierten Callbacks nicht erledigen lässt, können Sie auch selbst einen
# Callback programmieren. Callbacks werden durch Unterklassenbildung von
# keras.callbacks.Callback  implementiert.

import keras
import numpy as np

class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()