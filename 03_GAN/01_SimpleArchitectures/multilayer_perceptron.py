# example of a multilayer perceptron

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import pydot as pyd

visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)

output = Dense(1, activation='sigmoid')(hidden3)

model = Model(inputs=visible, outputs=output)

# summerize layers
model.summary()

#plot graph
plot_model(model, to_file='multilayer_perceptron_graph.png')

