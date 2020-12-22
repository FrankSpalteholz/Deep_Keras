# simple upsampling example

from numpy import asarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D

# define input data
X = asarray([[1,2],[3,4]])

# print data
print(X)
# reshape input data into one sample a sample with a channel
X = X.reshape((1,2,2,1))
#define model
model = Sequential()
model.add(UpSampling2D(input_shape=(2,2,1)))
#summerize the model
model.summary()
#make a prediction with the model
yhat=model.predict(X)
#reshape output to remove channel to make printing easier
yhat=yhat.reshape((4,4))
#output result
print(yhat)





