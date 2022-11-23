import tensorflow as tf
from glob import glob
import os

#@tf.function
def read_files(seq, slice_example_shape, slices_type):
    return tf.map_fn(
        lambda f: tf.io.parse_tensor(tf.io.read_file(f), slices_type), 
        seq, fn_output_signature=tf.TensorSpec(shape=slice_example_shape, dtype=slices_type))

def process_val_dataset(
  d, im_height, im_width, 
  slice_example_shape, slices_type, prefetch_num, image_norm):
  return d.map(
      lambda f: tf.reshape(tf.io.parse_tensor(tf.io.read_file(f), slices_type), slice_example_shape),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False
  ).map(
      lambda s: tf.image.resize(
          tf.expand_dims(tf.cast(s, tf.float32), -1), 
          [im_height, im_width],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
      ) * image_norm,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False
  ).prefetch(prefetch_num)

def process_and_batch_dataset(
    d, im_height, im_width, 
    batch_size, slice_example_shape,
    slices_type, seq_len, prefetch_num, image_norm):
    
    d = d.map(
        lambda seq: read_files(seq, slice_example_shape, slices_type),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).map(
        lambda seq: tf.image.resize(
            tf.expand_dims(tf.cast(seq, tf.float32), -1), 
            [im_height, im_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        ) * image_norm,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    #augmentation may go here

    if seq_len == 1:
      d = d.map(lambda seq: tf.squeeze(seq, axis=0))
    
    return d.batch(batch_size).prefetch(prefetch_num)

def read_dataset(
    input_dir,
    slices_type = tf.uint8,
    im_width = 1000,
    im_height = 1000,
    seq_len = 10, 
    batch_size = 4,
    test_split = 0.2,
    val_core_list = [],
    seed = 1000,
    image_norm = 1.0,
    prefetch_num = tf.data.AUTOTUNE,
    window_shift = 1,
    window_stride = 1):
    
    slice_example_shape = None

    cores_list = glob(os.path.join(input_dir, '*'))
    cores_list_traintest = list(filter(lambda c: os.path.basename(c) not in val_core_list, cores_list))
    cores_list_val = list(filter(lambda c: os.path.basename(c) in val_core_list, cores_list))
    
    traintest_dataset_len = 0
    traintest_datasets = []
    val_datasets = {}

    for core in cores_list_traintest:
        slices_list = glob(os.path.join(core, '*.slice'))
        slices_list.sort()
        slice_example_shape = slices_list[0]
        d = tf.data.Dataset.from_tensor_slices(slices_list)
        traintest_datasets.append(d)
        traintest_dataset_len += (d.cardinality() // seq_len)*seq_len
        
    slice_example_shape = tf.io.parse_tensor(tf.io.read_file(slice_example_shape), slices_type).shape
    
    for core in cores_list_val:
        slices_list = glob(os.path.join(core, '*.slice'))
        slices_list.sort()
        d = tf.data.Dataset.from_tensor_slices(slices_list)
        val_datasets[os.path.basename(core)] = d
        #traintest_dataset_len += (d.cardinality() // seq_len)*seq_len

    traintest_dataset = None
    for i in range(len(traintest_datasets)):
        traintest_datasets[i] = traintest_datasets[i] \
            .window(seq_len, shift=window_shift, stride=window_stride, drop_remainder=True) \
            .flat_map(lambda window: window.batch(seq_len))
        
        if i == 0:
            traintest_dataset = traintest_datasets[0]
        else:
            traintest_dataset = traintest_dataset.concatenate(traintest_datasets[i])
    
    traintest_dataset = traintest_dataset.shuffle(traintest_dataset_len, seed = seed, reshuffle_each_iteration=False)
    
    traintest_split = tf.cast(tf.cast(traintest_dataset_len, tf.float32)*(1-test_split), tf.int64)
    
    #print(traintest_split, slice_example_shape, slices_type)
    
    train_dataset = traintest_dataset.take(traintest_split).cache().shuffle(traintest_split, reshuffle_each_iteration=True)
    test_dataset = traintest_dataset.skip(traintest_split).cache()
    
    proc_data = lambda x: process_and_batch_dataset(
        x, im_height, im_width, 
        batch_size, slice_example_shape, slices_type,
        seq_len, prefetch_num, image_norm)

    proc_val_data = lambda x: process_val_dataset(
      x, im_height, im_width, 
      slice_example_shape, slices_type, prefetch_num,
      image_norm
    )
    
    train_dataset = proc_data(train_dataset)
    test_dataset = proc_data(test_dataset)

    val_datasets = {k: proc_val_data(v) for k, v in val_datasets.items()}
    
    return train_dataset, test_dataset, val_datasets