import tensorflow as tf
import numpy as np

NUM_QUANTIZERS = 16

class QuantizerBlock(tf.keras.layers.Layer):
    def __init__(self, codebook, *args, **kwargs):
        super(QuantizerBlock, self).__init__(*args, **kwargs)
        self.codebook = codebook

    @tf.function
    def call(self, inputs, **kwargs):
        inputs = tf.reshape(inputs, [1, 64])
        diff = tf.math.squared_difference(inputs, self.codebook)
        diff = tf.reduce_sum(diff, axis=-1)
        closest_codebook_idx = tf.argmin(diff, -1)
        closest_codebook_vector = tf.gather(self.codebook, closest_codebook_idx, axis=1)
        closest_codebook_vector = tf.reshape(closest_codebook_vector, (1, 1, 64))
        residual = inputs - closest_codebook_vector
        idx = tf.one_hot(closest_codebook_idx, 16)
        return idx, residual


class Quantizer(tf.keras.models.Model):
    def __init__(self, codebooks, *args, **kwargs):
        super(Quantizer, self).__init__(*args, **kwargs)
        self.codebooks = tuple(QuantizerBlock(codebook, name='output_0') for codebook in codebooks)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1, 64], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)], )
    def encode(self, input_frames, num_quantizers):
        return self.call(input_frames)

    @tf.function(input_signature=[tf.TensorSpec(shape=[46, 1, 1], dtype=tf.int32)])
    def decode(self, encoding_indices):
        return {'output_0': tf.ones([1, 1, 64], dtype=tf.float32)}

    def call(self, inputs, **kwargs):
        embds = []
        for codebook in self.codebooks:
            embd, inputs = codebook(inputs)
            embds.append(embd)

        res = tf.argmax(embds, -1, output_type=tf.int32)
        res = tf.concat([res, tf.ones([46 - NUM_QUANTIZERS, 1], dtype=tf.int32) * -1], axis=0)
        return {'output_0': tf.reshape(res, (46, 1, 1)), 'output_1': 4}


def load_model():
    quan = Quantizer(tuple(
        np.load(f'./weights/q{i}.npy') for i in range(1, NUM_QUANTIZERS + 1))
    )

    tf.saved_model.save(quan,
                        './quantizer_fp32',
                        signatures={'encode': quan.encode.get_concrete_function(),
                                    'decode': quan.decode.get_concrete_function()})
    res = quan(tf.ones([1, 1, 64]))
    print(res)
    quan.build([1, 1, 64])
    return quan


if __name__ == '__main__':
    quan = load_model()

    converter = tf.lite.TFLiteConverter.from_saved_model('./quantizer_fp32')
    tflitemodel = converter.convert()
    with open('./quantizer_fp32.tflite', 'wb') as f:
        f.write(tflitemodel)
