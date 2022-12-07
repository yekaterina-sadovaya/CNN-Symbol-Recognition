from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from pre_processing import crop

N_rotations = 4
N_images_for_1_label = 6000*N_rotations
N_labels = 10
N_samples_total = N_images_for_1_label*N_labels
root = 'train/'
root_test = 'test/'

n_classes = 10


# load train and test dataset
def load_dataset():
    # load dataset
    train_images = np.zeros([N_samples_total, 28, 28, 1])
    Array_of_labels = np.array([])
    for id in range(1, N_labels + 1):
        print('loading ... ' + str(id) + '/10 dataset ')
        Array_of_labels = np.append(Array_of_labels, np.ones(N_images_for_1_label) * (id - 1))
        for n in range(0, N_images_for_1_label):

            # Here I tried to add some pre-processing of the data, i.e. adding random rotation of pictures to
            # increase the number of training samples.
            # This is needed to read pictures correctly but it looks a bit ugly because I'm doing it manually.
            # Anyway, it's a test and my goal was to understand if these rotations are influencing the final results
            # at all. It turned out that they help to achieve a bit higher accuracy somehow
            if 6000*2 > n >= 6000:
                n_image = str((n - 6000) + 1)
            elif 6000*3 > n >= 6000*2:
                n_image = str((n - 6000*2) + 1)
            elif n >= 6000*3:
                n_image = str((n - 6000*3) + 1)
            else:
                n_image = str(n + 1)

            n_image = n_image.zfill(5)

            file = root + str(id) + '/' + n_image + '.jpg'
            img = load_img(file, color_mode="grayscale", target_size=(28, 28))
            if n >= 6000:
                # random rotation
                ALPHA = np.random.uniform(-90, 90)
                # fit parameters from data
                img = img.rotate(ALPHA)
            img_array = img_to_array(img)
            img_array = crop(img_array)
            train_images[n + N_images_for_1_label * (id - 1), :, :, :] = img_array

    # Read test data
    N_test = 1000
    test_images = np.zeros([N_test, 28, 28, 1])
    y_test = np.zeros(N_test)
    for n in range(0, N_test):
        n_image = str(n + 1)
        n_image = n_image.zfill(5)
        file = root + '1/' + n_image + '.jpg'
        img = load_img(file, color_mode="grayscale", target_size=(28, 28))
        test_images[n, :, :, :] = img_to_array(img)

    trainX = train_images
    trainY = Array_of_labels

    testX = test_images
    testY = y_test

    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY, n_classes)
    testY = to_categorical(testY, n_classes)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # define data preparation
    # datagen = ImageDataGenerator(zca_whitening=True)
    # # fit parameters from data
    # datagen.fit(train_norm)
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


# define cnn model
def define_model():
    model = Sequential()
    # 1 layer
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    # 2 layer
    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    # 3 layer
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# run the test harness for evaluating the model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    print('All data was loaded')
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    print('All pixels were processed')
    # define model
    model = define_model()
    print('Model was defined')

    # fit model
    model.fit(trainX, trainY, epochs=100, batch_size=128*2, validation_data=(testX, testY))
    # save model
    model.save('CNN_model.h5')


# run the test
run_test_harness()
