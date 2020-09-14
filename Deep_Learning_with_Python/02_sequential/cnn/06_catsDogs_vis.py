import numpy as np
from keras import models
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

os = 0;

model_win = load_model('models/cats_and_dogs_small_2.h5')
img_path_win= "D:\Dropbox\work\_SquareHouse\_CodeBase\dogs-vs-cats\cats_and_dogs_small" + r"\test\cats\cat.1700.jpg"

model_osx = load_model('models/cats_and_dogs_small_2.h5')
img_path_osx = '/Users/frankfurt/Dropbox/work/_SquareHouse/_CodeBase/dogs-vs-cats/cats_and_dogs_small/test/cats/cat.1700.jpg'

model = ''
img_path = ''

if os == 1:
    model = model_win
    img_path = img_path_win
elif os == 0:
    model = model_osx
    img_path = img_path_osx

model.summary()

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

# i = 0
# for i in range(32):
#     plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')
#     plt.show()
#     i += 1

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols,
                             images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std() + 1e-5
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0,
                                    255).astype('uint8')
            display_grid[col * size:
                            (col + 1) * size,
                            row * size:
                            (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()