import os
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow.python.framework import ops, dtypes

class AEROBIDataloader():
  def __init__(self, root_dir='/path/to/dataset', mode='train', num_classes=6):
    self.file_names = []
    self.num_classes = num_classes
    self.base_data_path = os.path.join(root_dir, mode)
    dirs = os.listdir(self.base_data_path)
    for directory in dirs:
      full_dir_path = os.path.join(self.base_data_path, directory)
      self.file_names.extend([os.path.join(full_dir_path, crop_name) for crop_name in os.listdir(full_dir_path)])

    self.total_images = len(self.file_names)

    dataset_xml_list = [os.path.join(root_dir, 'metadata/background.xml'),
                        os.path.join(root_dir, 'metadata/defects.xml')]

    self.target_filename_pairs = {}
    self.__init_filename_pairs(dataset_xml_list)
    self.hash_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(self.target_filename_pairs.keys(),
                                                  self.target_filename_pairs.values()), '000000')


  def __init_filename_pairs(self, xml_list):
    for file_name in xml_list:
      root_path, _ = os.path.splitext(os.path.basename(file_name))
      tree = ET.parse(file_name)
      root = tree.getroot()
      for defect in root:
        crop_name = defect.attrib.values()[0]
        target = self.__computeTarget(defect)
        self.target_filename_pairs[
          os.path.join(os.path.join(self.base_data_path, root_path), crop_name.split('.')[0])] = target
    return self.target_filename_pairs

  def __computeTarget(self, defect, ):
    out = ""
    for i in range(self.num_classes):
      out += defect[i].text + ','

    return out

  def get_labels(self, crop_paths):
    labels = []
    for image_path in crop_paths:
      f_dir_idx = image_path[:image_path.rfind('/')].rfind('/')
      delim = image_path.rfind('_-_')
      file_type = image_path.rfind('.')
      if delim != -1:
        crop_key = image_path[f_dir_idx + 1:delim] + image_path[file_type:]
      else:
        crop_key = image_path[f_dir_idx + 1:]
      labels.append(tf.constant(self.target_filename_pairs[crop_key]))
    return labels

  def __resize_img_width(self, image, initial_width, initial_height):
    ratio = tf.to_float(initial_height) / tf.constant(self.img_size, dtype=tf.float32)
    new_width = tf.cond(tf.equal(initial_width, initial_height),
                        lambda: tf.constant(self.img_size),
                        lambda: tf.to_int32(tf.to_float(initial_width) / ratio))
    new_height = tf.constant(self.img_size)
    return tf.image.resize_images(image, [new_width, new_height])

  def __resize_img_height(self, image, initial_width, initial_height):
    ratio = tf.to_float(initial_width) / tf.constant(self.img_size, dtype=tf.float32)
    new_width = tf.constant(self.img_size)
    new_height = tf.cond(tf.equal(initial_width, initial_height),
                         lambda: tf.constant(self.img_size),
                         lambda: tf.to_int32(tf.to_float(initial_height) / ratio))
    return tf.image.resize_images(image, [new_width, new_height])

  def __resize_image_keep_aspect(self, image):
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    result = tf.cond(tf.greater_equal(initial_width, initial_height),
                     lambda: self.__resize_img_width(image, initial_width, initial_height),
                     lambda: self.__resize_img_height(image, initial_width, initial_height))

    return result

  def __get_training_samples(self, file_queue):
    reader = tf.WholeFileReader()
    image_path, value = reader.read(file_queue)

    img = tf.image.decode_png(value, channels=3)

    resized_img = self.__resize_image_keep_aspect(img)
    crop = tf.random_crop(resized_img, [self.img_size, self.img_size, 3])
    flipped_cropped_img = tf.image.random_flip_left_right(crop)
    base = tf.string_split([image_path], '.').values
    crop_names = base[0]
    labels = tf.string_to_number(tf.string_split([self.hash_table.lookup(crop_names)], ',').values)
    return flipped_cropped_img, image_path, labels

  def get_batch_handle(self, img_size=224, batch_size=64, capacity=1100, min_after_dequeue=1000, num_epochs=310,
                       shuffle=True):
    self.img_size = img_size

    self.file_name_tensors = tf.convert_to_tensor(self.file_names, dtype=dtypes.string)
    self.file_queue = tf.train.string_input_producer(self.file_names, shuffle=True)

    self.sample_image, self.sample_image_path, self.lookup_values = self.__get_training_samples(self.file_queue)
    self.sample_image.set_shape((self.img_size, self.img_size, 3))
    self.lookup_values.set_shape((6))
    if shuffle:
      self.x_label, self.y_label = tf.train.shuffle_batch([self.sample_image, self.lookup_values],
                                                          batch_size=batch_size,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue,
                                                          allow_smaller_final_batch=True)
    else:
      self.x_label, self.y_label = tf.train.batch([self.sample_image, self.lookup_values],
                                                  batch_size=batch_size,
                                                  capacity=capacity,
                                                  allow_smaller_final_batch=True)

    """My new code"""
    self.x_label.set_shape((batch_size, self.img_size, self.img_size, 3))
    self.y_label.set_shape((batch_size, self.num_classes))

    return self.x_label, self.y_label


