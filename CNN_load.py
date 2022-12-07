# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from pre_processing import crop

root_test = 'test/'
N_test = 10000


def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run():
    # load model
    model = load_model('CNN_model.h5')
    model.summary()
    # load the image
    predictions = np.array([])
    for n in range(0, N_test):
        n_image = str(n)
        n_image = n_image.zfill(5)
        file = root_test + n_image + '.jpg'
        unknown_image = load_img(file, color_mode="grayscale", target_size=(28, 28))
        # convert to array
        unknown_image = img_to_array(unknown_image)
        # crop extra space around image
        unknown_image = crop(unknown_image)
        # reshape into a single sample with 1 channel
        unknown_image = unknown_image.reshape(1, 28, 28, 1)
        unknown_image = unknown_image.astype('float32')
        unknown_image = unknown_image / 255.0
        predict = np.argmax(model.predict(unknown_image))
        predictions = np.append(predictions, predict)

    return predictions


# entry point, run the example
pred = run()
with open("submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    for idx in range(N_test):
        CAT = pred[idx] + 1
        fp.write(f"{idx:05},{int(CAT)}\n")
