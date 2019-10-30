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


def DirectOutputFunction(inputs, outputs):
    return outputs


def DirectLossFunctionGenerator(inputs):
    """Direct loss
    See Section 3.2(1) for more details

    Args:
        inputs: A input batch(no use)

    Returns:
        direct architecture loss function
    """
    def loss_function(y_true, y_pred):
        return tf.reduce_sum(
            tf.square(range_compress(y_true) - range_compress(y_pred)))
    return loss_function


def WEOutputFunction(inputs, outputs):
    img1 = inputs[:, :, :, 9:12]
    img2 = inputs[:, :, :, 12:15]
    img3 = inputs[:, :, :, 15:18]
    weight1 = outputs[:, :, :, 0:3]
    weight2 = outputs[:, :, :, 3:6]
    weight3 = outputs[:, :, :, 6:9]
    total_weights = weight1 + weight2 + weight3 + WEIGHT_EPS
    blended = (img1 * weight1 + img2 * weight2 +
               img3 * weight3) / total_weights
    return blended


def WELossFunctionGenerator(inputs):
    """Weighted Estimator(WE) loss
    See Section 3.2(2) for more details

    Args:
        inputs: A input batch

    Returns:
        WE architecture loss function
    """
    def loss_function(y_true, y_pred):
        blended = WEOutputFunction(inputs, y_pred)
        return tf.reduce_sum(
            tf.square(range_compress(y_true) - range_compress(blended)))
    return loss_function


def WIEOutputFunction(inputs, outputs):
    img1 = outputs[:, :, :, 9:12]
    img2 = outputs[:, :, :, 12:15]
    img3 = outputs[:, :, :, 15:18]
    weight1 = outputs[:, :, :, 0:3]
    weight2 = outputs[:, :, :, 3:6]
    weight3 = outputs[:, :, :, 6:9]
    total_weights = weight1 + weight2 + weight3 + WEIGHT_EPS
    blended = (img1 * weight1 + img2 * weight2 +
               img3 * weight3) / total_weights
    return blended


def WIELossFunctionGenerator(inputs):
    """Weight and Image Estimator(WIE) loss
    See Section 3.2(3) for more details

    Args:
        inputs: A input batch(no use)

    Returns:
        WIE architecture loss function
    """
    def loss_function(y_true, y_pred):
        blended = WIEOutputFunction(inputs, y_pred)
        return tf.reduce_sum(
            tf.square(range_compress(y_true) - range_compress(blended)))
    return loss_function


def range_compress(img):
    """Differentiable tonemapping operator

    Args:
        img: input image/batch of images

    Returns:
        Tonemapped images
    """
    return tf.math.log(1.0 + MU * img) / tf.math.log(1.0 + MU)


def create_model_and_loss(model_type: str):
    """Create CNN model and corresponding loss according to type

    Args:
        model_type: str of "direct"/"we"/"wie"

    Returns:
        A tuple of
            1: CNN model with specific channels
            2: Corresponding loss function generator function
    """
    if model_type.lower() == "direct":
        return DHDRCNN(
            "direct"), DirectLossFunctionGenerator, DirectOutputFunction
    elif model_type.lower() == "we":
        return DHDRCNN("WE"), WELossFunctionGenerator, WEOutputFunction
    elif model_type.lower() == "wie":
        return DHDRCNN("WIE"), WIELossFunctionGenerator, WIEOutputFunction


def tf_compute_PSNR(inputs, reference):
    """Compute Peak signal-to-noise ratio(PSNR)

    Args:
        input: A produced image
        reference: A reference image

    Returns:
        Error in float
    """

    num_pixels = tf.size(inputs, out_type=tf.float32)
    squared_error = tf.reduce_sum(tf.square(inputs - reference)) / num_pixels
    numerator = tf.math.log(1.0 / squared_error)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    error = 10.0 * numerator / denominator
    return error
