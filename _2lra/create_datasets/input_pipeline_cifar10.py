# Refer to https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/image/input_pipeline.py

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_cifar10_datasets(n_devices, batch_size=256, normalize=False):
  """Get CIFAR-10 dataset splits."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  train_dataset = tfds.load('cifar10', split='train[:90%]')
  val_dataset = tfds.load('cifar10', split='train[90%:]')
  test_dataset = tfds.load('cifar10', split='test')

  def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }
    if normalize:
      decoded['inputs'] = decoded['inputs'] / 255
    return decoded

  train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

  # train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  train_dataset = train_dataset.shuffle(
      buffer_size=256, reshuffle_each_iteration=True)

  return train_dataset, val_dataset, test_dataset, 10, 256, (batch_size, 32, 32,
                                                             1)
