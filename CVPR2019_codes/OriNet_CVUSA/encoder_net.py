import tensorflow as tf
import ops

from tensorflow.python.ops.gen_math_ops import *

ngf = 64

class sem_encoder:

    def gen_conv(self, batch_input, out_channels):

        initializer = tf.random_normal_initializer(0, 0.02)

        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                    kernel_initializer=initializer)

    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(self, inputs):

        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


    def create_generator_gem_pooling_shallow_concatLast3conv(self, generator_inputs, name='sat_sem', trainable=True,
                                                             p=3.0, eps=1e-6):

        with tf.device('/gpu:0'):

            layers = []

            num_batch = tf.shape(generator_inputs)[0]
            height = tf.shape(generator_inputs)[1]
            width = tf.shape(generator_inputs)[2]

            # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.variable_scope(name + "encoder_1"):
                output = self.gen_conv(generator_inputs, ngf)
                layers.append(output)

            layer_specs = [
                ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ]

            for out_channels in layer_specs:
                with tf.variable_scope(name + "encoder_%d" % (len(layers) + 1)):
                    rectified = self.lrelu(layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = self.gen_conv(rectified, out_channels)
                    output = self.batchnorm(convolved)
                    layers.append(output)

            if trainable:
                # sample pixels corresponding to the last feature elements
                h, w = layers[-1].get_shape().as_list()[1:3]
                trace_locations = ops.trace_locations_backward
            else:
                # sample pixels corresponding to the whole image

                h, w = [height, width]
                trace_locations = ops.trace_locations_forward

            X, Y = tf.meshgrid(tf.range(w), tf.range(h), indexing='xy')
            loc_x = tf.tile(tf.reshape(X, [1, -1]), [num_batch, 1])
            loc_y = tf.tile(tf.reshape(Y, [1, -1]), [num_batch, 1])

            end_points = []
            end_points.append(layers[-3])
            end_points.append(layers[-2])
            end_points.append(layers[-1])

            locations = [trace_locations(loc_x, loc_y, [h, w], [tf.shape(feat)[1], tf.shape(feat)[2]])
                         for feat in end_points]

            net = ops.extract_values(end_points, locations)
            hyperchannels = tf.shape(net)[3]

            # hyperchannels = net.get_shape().as_list()[-1]

            net = tf.reshape(net, [num_batch, h, w, hyperchannels])

            h_relu = tf.maximum(net, tf.constant(eps))

            return tf.nn.l2_normalize(
                tf.pow(tf.reduce_mean(tf.pow(h_relu, tf.constant(p)), axis=[1, 2]), tf.constant(1.0 / p)), dim=1)
