import tensorflow as tf

keras = tf.keras
from keras import models, layers
import numpy as np
import ipdb


class ResLayer(tf.keras.layers.Layer):
    def __init__(self,
                 cin,
                 cout,
                 depth_dilation=1,
                 last_conv_groups=1,
                 verbose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=(3, 1), dilation_rate=(depth_dilation, 1))
        self.pointwise = layers.Conv2D(filters=cin, kernel_size=(1, 1))
        self.leaky_relu = layers.LeakyReLU()
        self.last_conv = layers.Conv2D(filters=cout, kernel_size=(1, 1), groups=last_conv_groups)
        self.verbose = verbose

    def call(self, inputs, **kwargs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.leaky_relu(x)
        if self.verbose:
            print(x)
        x = self.last_conv(x)
        return x


def quantize(x, quan_params):
    scale, offset = quan_params
    x = (x / scale) - offset
    x = tf.math.round(x)
    x = tf.cast(x, tf.int8)
    x = tf.cast(x, tf.float32)
    return x


def dequantize(x, quan_params):
    scale, offset = quan_params
    x = (x + offset) * scale
    return x


class LastQuantizedResLayer(tf.keras.layers.Layer):
    def __init__(self,
                 cin,
                 cout,
                 quantize_params,
                 dequantize_params,
                 depth_dilation=1,
                 last_conv_groups=1,
                 verbose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=(3, 1), dilation_rate=(depth_dilation, 1))
        self.pointwise = layers.Conv2D(filters=cin, kernel_size=(1, 1))
        self.leaky_relu = layers.LeakyReLU()
        self.last_conv = layers.Conv2D(filters=cout, kernel_size=(1, 1), groups=last_conv_groups)
        self.verbose = verbose
        self.quantize_params = quantize_params
        self.dequantize_params = dequantize_params

    def call(self, inputs, **kwargs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = quantize(x, self.quantize_params)
        x = self.leaky_relu(x)
        if self.verbose:
            print(x)
        x = self.last_conv(x)
        x = dequantize(x, self.dequantize_params)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, cin, first_concat, first_kernel_size=10, first_stride=5, first_conv_groups=1, last_conv_groups=1
                 , verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.first_conv = layers.Conv2D(filters=cin, kernel_size=(first_kernel_size, 1), strides=(first_stride, 1),
                                        groups=first_conv_groups)
        self.res1 = ResLayer(cin=cin, cout=cin, depth_dilation=1, last_conv_groups=last_conv_groups, verbose=verbose)
        self.res2 = ResLayer(cin=cin, cout=cin, depth_dilation=3, last_conv_groups=last_conv_groups)
        self.res3 = ResLayer(cin=cin, cout=cin, depth_dilation=9, last_conv_groups=last_conv_groups)
        self.first_concat = first_concat
        self.first_buffer = tf.Variable(tf.zeros(first_concat))
        self.res1_buffer = tf.Variable(tf.zeros((1, 2, 1, cin)))
        self.res2_buffer = tf.Variable(tf.zeros((1, 6, 1, cin)))
        self.res3_buffer = tf.Variable(tf.zeros((1, 18, 1, cin)))
        self.verbose = verbose

    @tf.function
    def _causal_pad(self, x, buffer, size):
        x = tf.concat([buffer, x], axis=1)
        next_padding = x[:, -size:, :, :]
        return x, next_padding

    def call(self, inputs, *args, **kwargs):
        # x, self.first_buffer = self._causal_pad(inputs, self.first_buffer, tf.constant(self.first_concat[1]))
        x, abd = self._causal_pad(inputs, self.first_buffer, tf.constant(self.first_concat[1]))
        self.first_buffer.assign(abd)
        # x = tf.concat([self.first_buffer, inputs], axis=1)
        # d = tf.Variable([0, -48, 0, 0], dtype=tf.int32)
        # e = tf.Variable([0, 0, 0, 1], dtype=tf.int32)
        # abc = tf.strided_slice(x, d, e)
        # self.first_buffer = x[:, tf.constant(-self.first_concat[1]):, :, :]
        res_inputs = self.first_conv(x)
        conv_input = tf.nn.leaky_relu(res_inputs, alpha=0.3)

        conv_input = tf.concat([self.res1_buffer, conv_input], axis=1)
        self.res1_buffer.assign(conv_input[:, -2:, :, :])

        res_output = self.res1(conv_input)
        res_inputs = tf.add(res_inputs, res_output)
        conv_input = tf.nn.leaky_relu(res_inputs, alpha=0.3)

        conv_input = tf.concat([self.res2_buffer, conv_input], axis=1)
        # self.res2_buffer = conv_input[:, -6:, :, :]
        self.res2_buffer.assign(conv_input[:, -6:, :, :])

        res_outputs = self.res2(conv_input)
        res_inputs = tf.add(res_outputs, res_inputs)
        conv_input = tf.nn.leaky_relu(res_inputs, alpha=0.3)

        conv_input = tf.concat([self.res3_buffer, conv_input], axis=1)
        # self.res3_buffer = conv_input[:, -18:, :, :]
        self.res3_buffer.assign(conv_input[:, -18:, :, :])

        res_outputs = self.res3(conv_input)
        res_inputs = tf.add(res_outputs, res_inputs)
        x = tf.nn.leaky_relu(res_inputs, alpha=0.3)
        return x


class QuantizedEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 cin,
                 first_concat,
                 first_kernel_size=10,
                 first_stride=5,
                 first_conv_groups=1,
                 last_conv_groups=1,
                 verbose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.first_conv = layers.Conv2D(filters=cin, kernel_size=(first_kernel_size, 1), strides=(first_stride, 1),
                                        groups=first_conv_groups)
        self.res1 = LastQuantizedResLayer(cin=cin,
                                          cout=cin,
                                          depth_dilation=1,
                                          last_conv_groups=last_conv_groups,
                                          quantize_params=(17.62967872619629, -67),
                                          dequantize_params=(6.528060436248779, -16),
                                          verbose=verbose)
        self.res2 = ResLayer(cin=cin,
                             cout=cin,
                             depth_dilation=3,
                             last_conv_groups=last_conv_groups)
        self.res3 = ResLayer(cin=cin,
                             cout=cin,
                             depth_dilation=9,
                             last_conv_groups=last_conv_groups)
        self.first_concat = first_concat
        self.first_buffer = tf.Variable(tf.zeros(first_concat))
        self.res1_buffer = tf.Variable(tf.zeros((1, 2, 1, cin)))
        self.res2_buffer = tf.Variable(tf.zeros((1, 6, 1, cin)))
        self.res3_buffer = tf.Variable(tf.zeros((1, 18, 1, cin)))
        self.verbose = verbose

    @tf.function
    def _causal_pad(self, x, buffer, size):
        x = tf.concat([buffer, x], axis=1)
        next_padding = x[:, -size:, :, :]
        return x, next_padding

    def call(self, inputs, *args, **kwargs):
        # x, self.first_buffer = self._causal_pad(inputs, self.first_buffer, tf.constant(self.first_concat[1]))
        x, abd = self._causal_pad(inputs, self.first_buffer, tf.constant(self.first_concat[1]))
        self.first_buffer.assign(abd)
        # x = tf.concat([self.first_buffer, inputs], axis=1)
        # d = tf.Variable([0, -48, 0, 0], dtype=tf.int32)
        # e = tf.Variable([0, 0, 0, 1], dtype=tf.int32)
        # abc = tf.strided_slice(x, d, e)
        # self.first_buffer = x[:, tf.constant(-self.first_concat[1]):, :, :]
        res_inputs = self.first_conv(x)
        conv_input = tf.nn.leaky_relu(res_inputs, alpha=0.3)

        conv_input = tf.concat([self.res1_buffer, conv_input], axis=1)
        self.res1_buffer.assign(conv_input[:, -2:, :, :])

        res_output = self.res1(conv_input)
        res_inputs = tf.add(res_inputs, res_output)
        res_inputs = quantize(res_inputs, (12.716455459594727, -23))
        conv_input = tf.nn.leaky_relu(res_inputs, alpha=0.3)
        conv_input = dequantize(conv_input, (5.548006534576416, 47))

        conv_input = tf.concat([self.res2_buffer, conv_input], axis=1)
        conv_input = quantize(conv_input, (5.548006534576416, 47))
        # self.res2_buffer = conv_input[:, -6:, :, :]
        temp_buf = conv_input[:, -6:, :, :]
        temp_buf = dequantize(temp_buf, (5.548006534576416, 47))
        self.res2_buffer.assign(temp_buf)

        res_outputs = self.res2(conv_input)
        res_inputs = tf.add(res_outputs, res_inputs)
        conv_input = tf.nn.leaky_relu(res_inputs, alpha=0.3)
        conv_input = dequantize(conv_input, (4.458680152893066, 38))

        conv_input = tf.concat([self.res3_buffer, conv_input], axis=1)
        conv_input = quantize(conv_input, (4.458680152893066, 38))
        # self.res3_buffer = conv_input[:, -18:, :, :]
        temp_buf = conv_input[:, -18:, :, :]
        temp_buf = dequantize(temp_buf, (4.458680152893066, 38))
        self.res3_buffer.assign(temp_buf)

        res_outputs = self.res3(conv_input)
        res_inputs = tf.add(res_outputs, res_inputs)
        x = tf.nn.leaky_relu(res_inputs, alpha=0.3)
        x = dequantize(x, (3.698859930038452, 34))
        return x


class SoundStreamEncoder(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc1 = Encoder(cin=64, first_kernel_size=64, first_stride=16, first_concat=(1, 48, 1, 1),
                            last_conv_groups=1)
        self.enc2 = Encoder(cin=128, first_kernel_size=10, first_stride=5, first_concat=(1, 5, 1, 64),
                            last_conv_groups=2)
        self.enc3 = Encoder(cin=256, first_kernel_size=4, first_stride=2, first_concat=(1, 2, 1, 128),
                            last_conv_groups=4, first_conv_groups=2, verbose=False)
        self.conv1 = layers.Conv2D(filters=512, kernel_size=(4, 1), strides=(2, 1), groups=4)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 1), groups=4)

        self.first_buffer = tf.Variable(tf.zeros((1, 2, 1, 256)))
        self.second_buffer = tf.Variable(tf.zeros((1, 2, 1, 512)))

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, (1, 320, 1, 1))
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = tf.concat([self.first_buffer, x], axis=1)
        self.first_buffer.assign(x[:, -2:, :, :])
        # self.first_buffer = x[:, -2:, :, :]
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = tf.concat([self.second_buffer, x], axis=1)
        self.second_buffer.assign(x[:, -2:, :, :])
        # self.second_buffer = x[:, -2:, :, :]
        x = self.conv2(x)
        x = tf.squeeze(x, 1)
        return x


class SoundStreamEncoderQuantized(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc1 = Encoder(cin=64, first_kernel_size=64, first_stride=16, first_concat=(1, 48, 1, 1),
                            last_conv_groups=1)
        self.enc2 = Encoder(cin=128, first_kernel_size=10, first_stride=5, first_concat=(1, 5, 1, 64),
                            last_conv_groups=2)
        self.enc3 = QuantizedEncoder(cin=256, first_kernel_size=4, first_stride=2, first_concat=(1, 2, 1, 128),
                                     last_conv_groups=4, first_conv_groups=2, verbose=False)
        self.conv1 = layers.Conv2D(filters=512, kernel_size=(4, 1), strides=(2, 1), groups=4)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 1), groups=4)

        self.first_buffer = tf.Variable(tf.zeros((1, 2, 1, 256)))
        self.second_buffer = tf.Variable(tf.zeros((1, 2, 1, 512)))

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, (1, 320, 1, 1))
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = tf.concat([self.first_buffer, x], axis=1)
        x = quantize(x, (3.698859930038452, 34))
        temp_buf = x[:, -2:, :, :]
        temp_buf = dequantize(temp_buf, (3.698859930038452, 34))
        self.first_buffer.assign(temp_buf)
        # self.first_buffer = x[:, -2:, :, :]
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = dequantize(x, (1.0672332048416138, 38))
        x = tf.concat([self.second_buffer, x], axis=1)
        x = quantize(x, (1.0672332048416138, 38))
        temp_buf = x[:, -2:, :, :]
        temp_buf = dequantize(temp_buf, (1.0672332048416138, 38))
        self.second_buffer.assign(temp_buf)
        # self.second_buffer = x[:, -2:, :, :]
        x = self.conv2(x)
        x = tf.squeeze(x, 1)
        x = dequantize(x, (0.26349151134490967, -20))
        return x


# model = keras.models.Sequential([
#     Encoder(cin=64, first_kernel_size=64, first_stride=16, concat_shapes=[2, 6, 18], first_concat=48, last_conv_groups=1),
#     Encoder(cin=128, first_kernel_size=10, first_stride=5, concat_shapes=[2, 6, 18], first_concat=5, last_conv_groups=2),
#     Encoder(cin=256, first_kernel_size=4, first_stride=2, concat_shapes=[2, 6, 18], first_concat=2, last_conv_groups=4, first_conv_groups=2),
#     Concat(2),
#     layers.Conv2D(filters=512, kernel_size=(4, 1), strides=(2, 1), groups=4),
#     layers.LeakyReLU(),
#     Concat(2),
#     layers.Conv2D(filters=64, kernel_size=(3, 1), groups=4),
# ])


def try_set_weights(root_obj, weights, bias):
    root_obj.set_weights([
        np.float32(weights).transpose((1, 2, 3, 0)),
        np.float32(bias)
    ])
    return

    transposes = [
        # (3, 2, 1, 0),
        # (1, 0, 3, 2),
        # (2, 1, 0, 3),
        # (0, 3, 2, 1),
        # (3, 0, 1, 2),
        (1, 2, 3, 0)
    ]

    for tran in transposes:
        error = None
        try:
            root_obj.set_weights([
                weights.transpose(tran),
                bias
            ])
            # print(f'found match {root_obj} -> {tran}')
            return

        except ValueError as e:
            error = e
            continue

    print(root_obj)
    print(weights.shape)
    print(root_obj.weights[0].shape)
    raise error


def load_weights(model):
    for enc in ['enc1',
                'enc2',
                'enc3',
                ]:
        try_set_weights(getattr(model, enc).first_conv,
                        np.load(f'weights/{enc}.first_conv.weights.npy'),
                        np.load(f'weights/{enc}.first_conv.bias.npy'))

        for res in ['res1', 'res2', 'res3']:
            current_enc = getattr(model, enc)
            current_res = getattr(current_enc, res)

            # weight = np.load(f'weights\\{enc}.{res}.depthwise.weights.npy')
            # print(weight.max(), weight.min())
            #
            # weight = np.load(f'weights\\{enc}.{res}.pointwise.weights.npy')
            # print(weight.max(), weight.min())
            #
            # weight = np.load(f'weights\\{enc}.{res}.last_conv.weights.npy')
            # print(weight.max(), weight.min())

            try_set_weights(current_res.depthwise,
                            np.load(f'weights/{enc}.{res}.depthwise.weights.npy'),
                            np.load(f'weights/{enc}.{res}.depthwise.bias.npy'))
            try_set_weights(current_res.pointwise,
                            np.load(f'weights/{enc}.{res}.pointwise.weights.npy'),
                            np.load(f'weights/{enc}.{res}.pointwise.bias.npy'))
            try_set_weights(current_res.last_conv,
                            np.load(f'weights/{enc}.{res}.last_conv.weights.npy'),
                            np.load(f'weights/{enc}.{res}.last_conv.bias.npy'))

    try_set_weights(model.conv1,
                    np.load(f'weights/conv1.weights.npy'),
                    np.load(f'weights/conv1.bias.npy'), )
    try_set_weights(model.conv2,
                    np.load(f'weights/conv2.weights.npy'),
                    np.load(f'weights/conv2.bias.npy'), )
    return model


def load_weights_quantized(model):
    for enc in ['enc1',
                'enc2',
                # 'enc3',
                ]:
        try_set_weights(getattr(model, enc).first_conv,
                        np.load(f'weights/{enc}.first_conv.weights.npy'),
                        np.load(f'weights/{enc}.first_conv.bias.npy'))

        for res in ['res1', 'res2', 'res3']:
            current_enc = getattr(model, enc)
            current_res = getattr(current_enc, res)

            try_set_weights(current_res.depthwise,
                            np.load(f'weights/{enc}.{res}.depthwise.weights.npy'),
                            np.load(f'weights/{enc}.{res}.depthwise.bias.npy'))
            try_set_weights(current_res.pointwise,
                            np.load(f'weights/{enc}.{res}.pointwise.weights.npy'),
                            np.load(f'weights/{enc}.{res}.pointwise.bias.npy'))
            try_set_weights(current_res.last_conv,
                            np.load(f'weights/{enc}.{res}.last_conv.weights.npy'),
                            np.load(f'weights/{enc}.{res}.last_conv.bias.npy'))

    # enc 3
    enc = 'enc3'
    try_set_weights(getattr(model, enc).first_conv,
                    np.load(f'weights/{enc}.first_conv.weights.npy'),
                    np.load(f'weights/{enc}.first_conv.bias.npy'))

    # res 1
    res = 'res1'
    current_enc = getattr(model, enc)
    current_res = getattr(current_enc, res)



    try_set_weights(current_res.depthwise,
                    np.load(f'weights/{enc}.{res}.depthwise.weights.npy'),
                    np.load(f'weights/{enc}.{res}.depthwise.bias.npy'))
    try_set_weights(current_res.pointwise,
                    np.load(f'weights/{enc}.{res}.pointwise.weights.npy'),
                    np.load(f'weights/{enc}.{res}.pointwise.bias.npy'))
    try_set_weights(current_res.last_conv,
                    np.load(f'weights/{enc}.{res}.last_conv_weights_q.npy'),
                    np.load(f'weights/{enc}.{res}.last_conv_bias_q.npy'))

    for res in ['res2', 'res3']:
        current_res = getattr(current_enc, res)

        try_set_weights(current_res.depthwise,
                        np.load(f'weights/{enc}.{res}.depthwise_weights_q.npy'),
                        np.load(f'weights/{enc}.{res}.depthwise_bias_q.npy'))
        try_set_weights(current_res.pointwise,
                        np.load(f'weights/{enc}.{res}.pointwise_weights_q.npy'),
                        np.load(f'weights/{enc}.{res}.pointwise_bias_q.npy'))
        try_set_weights(current_res.last_conv,
                        np.load(f'weights/{enc}.{res}.last_conv_weights_q.npy'),
                        np.load(f'weights/{enc}.{res}.last_conv_bias_q.npy'))

    try_set_weights(model.conv1,
                    np.load(f'weights/conv1.weights_q.npy'),
                    np.load(f'weights/conv1.bias_q.npy'), )
    try_set_weights(model.conv2,
                    np.load(f'weights/conv2.weights_q.npy'),
                    np.load(f'weights/conv2.bias_q.npy'), )
    return model


def test_model(model):
    inter = tf.lite.Interpreter('./soundstream_encoder.tflite')
    inter.allocate_tensors()
    tflite_model = inter.get_signature_runner()
    x = np.ones((1, 320), dtype=np.float32)
    res = tflite_model(input_audio=x)['output_0']

    my_res = model(x)

    print(my_res)
    print(res)
    print(np.allclose(res, my_res))
    # import ipdb;
    # ipdb.set_trace(context=20)


def load_model(input_shape=(1, 320)):
    model = SoundStreamEncoder()
    model.build(input_shape)
    # model.summary()
    res = model(tf.zeros(input_shape))

    model = load_weights(model)
    return model

def load_quantized_model(input_shape=(1, 320)):
    model = SoundStreamEncoderQuantized()
    model.build(input_shape)
    # model.summary()
    res = model(tf.zeros(input_shape))

    model = load_weights_quantized(model)
    return model

def representative_dataset_gen():
    for i in range(100):
        yield [np.random.random((1, 320)).astype(np.float32)]


if __name__ == '__main__':
    # block(tf.zeros([1, 2, 3, 3]))

    # block = layers.Conv2D(filters=64, kernel_size=(64, 1), strides=(16, 1))
    # res = ResLayer(cin=64, cout=64, concat_shape=2)
    # block.build([1, 368, 1, 1])
    # res.build(([1, 20, 1, 64], [1, 22, 1, 64]))
    # result = block(tf.zeros([1, 368, 1, 1]))
    # # depth_in = tf.concat([tf.zeros([1, 2, 1, 64]), result], axis=1)
    # result = res(result, depth_in)
    # model = load_quantized_model()
    model = load_model()

    model.save('my_model', model)
    # fp32 tflite
    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp16_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp16_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    fp16_converter.target_spec.supported_types = [tf.float16]
    tflitemodel = fp16_converter.convert()
    with open('my_model_fp16.tflite', 'wb') as f:
        f.write(tflitemodel)
    tflite_model = fp32_converter.convert()
    with open('my_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # int8 tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    #     tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflitemodel = converter.convert()
    with open('my_model_int8.tflite', 'wb') as f:
        f.write(tflitemodel)

    test_model(model)

    # model = SoundstreamEncoder
    # model.build((1, 368, 1, 1))
    # model.summary()
