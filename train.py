import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras_preprocessing.image import ImageDataGenerator
from skimage import color
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
import seaborn as sns

_DEBUG = True

# resize image to this size
image_size = 200

# paths to data
train_data_path = "./train/"
test_data_path = "./test/"
validation_data_path = "./validation/"

labels = ['PNEUMONIA', 'NORMAL']

# image arrays
test_images = []
train_images = []

learning_history = None

num_classes = 0

X_train, y_train, X_test, y_test = [], [], [], []

# added early stopping, so the network won't be overfitting
callback = [
    EarlyStopping(monitor='loss', patience=7),
    ReduceLROnPlateau(monitor='loss', patience=4),
    ModelCheckpoint('./models/model.hdf5', monitor='loss', save_best_only=True)  # saving the best model
]

# generator for the image dataset -> better results of training
data_gen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    horizontal_flip=False,
    vertical_flip=False,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15
)


# loading data to arrays used in building the network
def load_data(dir_path, array):
    for label in labels:
        path = dir_path + label
        for image in os.listdir(path):
            if image is not None:
                image_path = os.path.join(path, image)
                # resizing images to 200x200px, the network will be calculated faster
                image = cv2.resize(cv2.imread(image_path), (image_size, image_size))
                array.append([image, label])


# shuffle the arrays, because the data goes through the network in batches
def shuffle_arrays():
    for i in range(8):
        np.random.shuffle(train_images)
        np.random.shuffle(test_images)


# show example images of the dataset on a plot
def show_example_images(array):
    example = plt
    for index in range(15):
        example.subplot(3, 5, index + 1)
        example.imshow(array[index][0])
        example.title("{}".format(array[index][1]))

    example.show()


# 1 if has pneumonia, 0 if not
def has_pneumonia(label):
    if label == 'NORMAL':
        return 0
    else:
        return 1


# split the data to 2 variables (X and Y for the network)
def split_data(array):
    x = []
    y = []
    for i, (value, label) in enumerate(array):
        x.append(value)
        y.append(has_pneumonia(label))
    return np.array(x), np.array(y)


# applying grayscale to the images
def preprocess_data(data):
    ret_data = color.rgb2gray(data).reshape(-1, image_size, image_size, 1).astype('float32')
    ret_data /= 255
    return ret_data


# here is the model saved as Sequential
def get_model(num_classes):
    return Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 1)),
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Flatten(),

        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation="softmax")

    ])


# draws a diagram of the network's learning over time
def draw_learning_curve(history, keys=None):
    if keys is None:
        keys = ['accuracy', 'loss']
    plt.figure(figsize=(20, 8))
    for i, key in enumerate(keys):
        plt.subplot(1, 2, i + 1)
        sns.lineplot(x=history.epoch, y=history.history[key])
        sns.lineplot(x=history.epoch, y=history.history['val_' + key])
        plt.title('Learning Curve')
        plt.ylabel(key.title())
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='best')
    plt.show()


# training the network
def train_network():
    print("Training network")

    data_gen.fit(X_train)
    train_gen = data_gen.flow(X_train, y_train, batch_size=32)

    model = get_model(num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    learning_history = model.fit_generator(
        train_gen,
        epochs=100,
        steps_per_epoch=X_train.shape[0] // 32,
        validation_data=(X_test, y_test),
        callbacks=callback,
    )
    print("Network trained!")

    draw_learning_curve(learning_history)


# checks the real score of the trained network
def evaluate_network(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}%'.format(score[0] * 100))
    print('Test accuracy: {}%'.format(score[1] * 100))
    return score


# prepares the dataset for further operations
def prepare_data():
    print("Loading train images...")
    load_data(train_data_path, train_images)
    print("Loaded", len(train_images), " train images.")
    print("Loading test images...")
    load_data(test_data_path, test_images)
    print("Loaded", len(test_images), " test images.")
    print("Finished loading data")

    shuffle_arrays()
    if _DEBUG:
        show_example_images(train_images)

    X_train, y_train = split_data(train_images)
    X_test, y_test = split_data(test_images)

    # preprocess
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    num_classes = y_train.shape[1]

    train_network()

    model = get_model(num_classes)
    model.summary()

    evaluate_network(model=model, x_test=X_test, y_test=y_test)
