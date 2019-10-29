import tensorflow as tf
from config import MU, WEIGHT_EPS

output_shape = {'direct': 3, 'WE': 9, 'WIE': 18}


class DHDRCNN(tf.keras.Model):

    def __init__(self, model_type):
        super(DHDRCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(100, (7, 7), activation='relu', strides=(
            1, 1), padding='same', kernel_initializer='zeros')
        self.conv2 = tf.keras.layers.Conv2D(100, (5, 5), activation='relu', strides=(
            1, 1), padding='same', kernel_initializer='zeros')
        self.conv3 = tf.keras.layers.Conv2D(50, (3, 3), activation='relu', strides=(
            1, 1), padding='same', kernel_initializer='zeros')
        self.conv4 = tf.keras.layers.Conv2D(output_shape[model_type], (1, 1), activation='sigmoid', strides=(
            1, 1), padding='same', kernel_initializer='zeros')

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return self.conv4(x3)


def DirectLossFunctionGenerator(inputs):
    def loss_function(y_true, y_pred):
        return tf.reduce_sum(
            tf.square(range_compress(y_true) - range_compress(y_pred)))
    return loss_function


def WELossFunctionGenerator(inputs):
    def loss_function(y_true, y_pred):
        img1 = inputs[:, :, :, 9:12]
        img2 = inputs[:, :, :, 12:15]
        img3 = inputs[:, :, :, 15:18]
        weight1 = y_pred[:, :, :, 0:3]
        weight2 = y_pred[:, :, :, 3:6]
        weight3 = y_pred[:, :, :, 6:9]
        total_weights = weight1 + weight2 + weight3 + WEIGHT_EPS
        blended = (img1 * weight1 + img2 * weight2 + img3 * weight3) / total_weights
        return tf.reduce_sum(
            tf.square(range_compress(y_true) - range_compress(blended)))
    return loss_function


def WIELossFunctionGenerator(inputs):
    def loss_function(y_true, y_pred):
        img1 = y_pred[:, :, :, 9:12]
        img2 = y_pred[:, :, :, 12:15]
        img3 = y_pred[:, :, :, 15:18]
        weight1 = y_pred[:, :, :, 0:3]
        weight2 = y_pred[:, :, :, 3:6]
        weight3 = y_pred[:, :, :, 6:9]
        total_weights = weight1 + weight2 + weight3 + WEIGHT_EPS
        blended = (img1 * weight1 + img2 * weight2 + img3 * weight3) / total_weights
        return tf.reduce_sum(
            tf.square(range_compress(y_true) - range_compress(blended)))
    return loss_function


def range_compress(img):
    return tf.math.log(1.0 + MU * img) / tf.math.log(1.0 + MU)


def create_model_and_loss(model_type: str):
    if model_type.lower() == "direct":
        return DHDRCNN("direct"), DirectLossFunctionGenerator
    elif model_type.lower() == "we":
        return DHDRCNN("WE"), WELossFunctionGenerator
    elif model_type.lower() == "wie":
        return DHDRCNN("WIE"), WIELossFunctionGenerator


def tf_compute_PSNR(input, reference):
    """Compute Peak signal-to-noise ratio(PSNR)

    Args:
        input: A produced image
        reference: A reference image

    Returns:
        Error in float
    """

    num_pixels = input.size
    squared_error = tf.reduce_sum(tf.square(input - reference)) / num_pixels
    numerator = tf.math.log(1.0 / squared_error)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    error = 10.0 * numerator / denominator
    return error
