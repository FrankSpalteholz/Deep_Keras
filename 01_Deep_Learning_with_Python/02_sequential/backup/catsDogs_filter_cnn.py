from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# loading pre-trained model
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output

# loss = K.mean(layer_output[:, :, :, filter_index])
#
# # calculation of gradient of the loss-functions from filter 0 of layer block3-conv1
# grads = K.gradients(loss, model.input)[0]
#
# # normalizing the gradients
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
# # create new numpy-list with 2 tensors (value of loss and value of gradient)
# iterate = K.function([model.input], [loss, grads])
# loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
#
# input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
# step = 1.

# # maximizing stochastic gradient descent
# for i in range(40):
#     loss_value, grads_value = iterate([input_img_data])
#     input_img_data += grads_value * step
#

# convert new tensor-data to image-like-data


def deprocess_image(x):

    # normalizing tensor, mean(0), standart diviation 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clipping to interval between 0 and 1
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to rgb array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# function that takes a layer-name and filter-index
# returning image-tensor showing the pattern of the filter-maximization

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output

    # create/define loss function for max the activation of the desired layer
    loss = K.mean(layer_output[:, :, :, filter_index])

    # calc gradient of loss function of the input
    grads = K.gradients(loss, model.input)[0]

    # normalization of the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # put loss and gradient of the input into a list
    iterate = K.function([model.input], [loss, grads])

    # create new gray image with some random noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.

    # calc gradient descent 40 times
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        img = input_img_data[0]
    return deprocess_image(img)



plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

layer_name = 'block4_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin,
                    8 * size + 7 * margin, 3))

for i in range(8):

    for j in range(8):

        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        #filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        #print(horizontal_start, horizontal_end, vertical_start, vertical_end)
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
        #results[0:64,0:64,:] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results.astype('uint8'))
plt.show()