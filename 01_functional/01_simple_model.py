from keras import layers
from keras import Input, Model
import numpy as np

# 1. define the model,
# 2. compile the model
# 3. fit the model to data
# 4. evaluate model 

input_tensor = Input(shape=(64,))

x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)

output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor,output_tensor)
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128)
score=model.evaluate(x_train, y_train)