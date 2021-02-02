from tensorflow.keras.models import load_model
from tensorflow.python.keras.distribute.distribute_strategy_test import get_model
import cv2
from skimage import color
from tensorflow.python.keras.models import Sequential

# resize image to this size
image_size = 200
num_classes = 0


# predicts the result of the network, returns the class (without probability)
def predict(model: Sequential, sample):
    prediction = model.predict_classes(sample)
    return prediction


# applying grayscale to the images
def preprocess_data(data):
    ret_data = color.rgb2gray(data).reshape(-1, image_size, image_size, 1).astype('float32')
    ret_data /= 255
    return ret_data


# resizing the given image for prediction
def process_sample(sample_path):
    image = cv2.resize(cv2.imread(sample_path), (image_size, image_size))
    image = preprocess_data(image)
    return image


# loads existing network for further use
def load_network_model(network_model):
    print("Loading model...")
    network_model = load_model('model.hdf5')
    print("Model loaded!")
    return network_model


if __name__ == "__main__":
    model = get_model()
    model = load_network_model(model)

    sample = process_sample("./val/PNEUMONIA/person1954_bacteria_4886.jpeg")
    prediction_class = predict(model, sample)

    if prediction_class is 0:
        print("Sample predicted class: NORMAL")
    else:
        print("Sample predicted class: PNEUMONIA")
