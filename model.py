import tensorflow as tf

MU = 5000
output_shape = {'direct': 3, 'WE': 9, 'WIE': '18'}

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


class DirectLossFunction(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.square(range_compress(y_true) - range_compress(y_pred)))

class WELossFunction(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        pass

class WIELossFunction(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        pass


def range_compress(img):
    return tf.math.log(1.0 + MU * img) / tf.math.log(1.0 + MU)


def create_model_and_loss(model_type: str):
    if model_type.lower() == "direct":
        return DHDRCNN("direct"), DirectLossFunction()
    elif model_type.lower() == "we":
        return DHDRCNN("WE"), WELossFunction()
    elif model_type.lower() == "wie":
        return DHDRCNN("WIE"), WIELossFunction()