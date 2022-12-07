import numpy as np
from skimage.transform import resize


def crop(image):
    # Remove empty space from top
    available = True
    while(available):
        line = image[0,:]
        available = True
        for i in range(len(line)):
            if line[i] > 4:
                available = False
        if available:
            image = np.delete(image, 0,0)

    # Remove empty space from bottom
    available = True
    while(available):
        line = image[-1,:]
        available = True
        for i in range(len(line)):
            if line[i] > 4:
                available = False
        if available:
            image = np.delete(image, -1,0)

    # Remove empty space from left
    available = True
    while(available):
        line = image[:,0]
        available = True
        for i in range(len(line)):
            if line[i] > 4:
                available = False
        if available:
            image = np.delete(image, 0,1)

    # Remove empty space from right
    available = True
    while(available):
        line = image[:,-1]
        available = True
        for i in range(len(line)):
            if line[i] > 4:
                available = False
        if available:
            image = np.delete(image, -1,1)
    image = resize(image, (28, 28))
    return image
