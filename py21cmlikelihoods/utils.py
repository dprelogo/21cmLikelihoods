import numpy as np
import tensorflow as tf

def _prepare_dataset_Gauss(data_samples, param_samples, batch_size):
    N = len(data_samples)
    data_samples = tf.data.Dataset.from_tensor_slices(data_samples)
    param_samples = tf.data.Dataset.from_tensor_slices(param_samples)

    total = tf.data.Dataset.zip((param_samples, data_samples))
    return total.shuffle(N).batch(batch_size)

def _prepare_dataset_CMAF(data_samples, param_samples, batch_size):
    N = len(data_samples)
    dummy_zeros = tf.data.Dataset.from_tensor_slices(np.zeros((N,), dtype=data_samples.dtype))
    data_samples = tf.data.Dataset.from_tensor_slices(data_samples)
    param_samples = tf.data.Dataset.from_tensor_slices(param_samples)
    
    total = tf.data.Dataset.zip((data_samples, param_samples))
    return tf.data.Dataset.zip((total, dummy_zeros)).shuffle(N).batch(batch_size)

def prepare_dataset(NDE, data_samples, param_samples, batch_size):
    from . import (
        ConditionalGaussian, 
        ConditionalGaussianMixture, 
        ConditionalMaskedAutoregressiveFlow,
    )
    
    if isinstance(NDE, ConditionalMaskedAutoregressiveFlow):
        return _prepare_dataset_CMAF(data_samples, param_samples, batch_size)
    elif isinstance(NDE, (ConditionalGaussian, ConditionalGaussianMixture)):
        return _prepare_dataset_Gauss(data_samples, param_samples, batch_size)
    else:
        raise TypeError(
            "NDE should be `ConditionalGaussian`, `ConditionalGaussianMixture` "
            "or `ConditionalMaskedAutoregressiveFlow."
        )
    
def check_callable(f, *args, **kwargs):
    if callable(f):
        return f(*args, **kwargs)
    else:
        return f