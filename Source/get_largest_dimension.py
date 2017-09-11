import os
from PIL import Image


largest = 0
for directory in os.listdir("../Images"):
    print(directory + '\r', end='')
    for image in os.listdir("../Images/" + directory):
        try:
            print(image + '\r', end='')
            img = Image.open("../Images/" + directory + '/' + image)
            width, height = img.size
            if width >= largest:
                largest = width
            if height >= largest:
                largest = height
        except:
            pass
print()
print('Largest Dimension of any file in set: {}'.format(largest))
