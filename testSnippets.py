import numpy as np
import simpleaudio as sa

matrix = np.zeros((2,3,5))

print(matrix)

filename = '/Users/frankfurt/Desktop/Applaus.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing
