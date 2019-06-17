import os
import sys

import numpy as np
import tensorflow as tf

from src.AEROBI.image_ops import conv
from src.AEROBI.image_ops import fully_connected
from src.AEROBI.image_ops import batch_norm
from src.AEROBI.image_ops import relu
from src.AEROBI.image_ops import max_pool
from src.AEROBI.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.AEROBI.data_utils import AEROBIDataloader

class Model(object):
  def __init__(self,
               cutout_size=None,
               batch_size=32,
               eval_batch_size=50,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.1,
               keep_prob=1.0,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               name="generic_model",
               seed=None,
               num_epochs=None,
               rl_steps=None
              ):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    print "-" * 80
    print "Build model {}".format(name)
    self.train_dataset_handler = AEROBIDataloader(mode='train')
    self.val_dataset_handler = AEROBIDataloader(mode='val')
    self.val_rl_dataset_handler = AEROBIDataloader(mode='val')
    self.test_dataset_handler = AEROBIDataloader(mode='test')


    self.cutout_size = cutout_size
    self.batch_size = batch_size
    self.eval_batch_size = eval_batch_size
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.l2_reg = l2_reg
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_rate = lr_dec_rate
    self.keep_prob = keep_prob
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.data_format = data_format
    self.name = name
    self.seed = seed
    
    self.global_step = None
    self.valid_acc = None
    self.test_acc = None
    self.num_epochs = num_epochs

    print "Build data ops"
    with tf.device("/cpu:0"):

      # training data
      self.num_train_examples = self.train_dataset_handler.total_images
      self.num_train_batches = (
        self.num_train_examples + self.batch_size - 1) // self.batch_size

      self.lr_dec_every = lr_dec_every * self.num_train_batches

      self.x_train, self.y_train = self.train_dataset_handler.get_batch_handle(batch_size=self.batch_size, num_epochs=self.num_epochs)
      self.x_valid, self.y_valid = self.val_dataset_handler.get_batch_handle(batch_size=self.eval_batch_size, num_epochs=self.num_epochs, shuffle=False)

      batches_per_epoch_rl = self.val_rl_dataset_handler.total_images / (1.0* eval_batch_size )

      epochs_for_rl = self.num_epochs * rl_steps/(1.0*batches_per_epoch_rl)

      self.x_valid_rl, self.y_valid_rl = self.val_dataset_handler.get_batch_handle(batch_size=self.eval_batch_size, num_epochs=epochs_for_rl, shuffle=True)


      self.num_valid_examples = self.val_dataset_handler.total_images
      self.num_valid_batches = (
          (self.num_valid_examples + self.eval_batch_size - 1)
          // self.eval_batch_size)

      self.num_test_examples = self.test_dataset_handler.total_images
      self.num_test_batches = (
        (self.num_test_examples + self.eval_batch_size - 1)
        // self.eval_batch_size)
      self.x_test, self.y_test = self.test_dataset_handler.get_batch_handle(batch_size=self.eval_batch_size, num_epochs=self.num_epochs, shuffle=False)

      if self.data_format == "NCHW":
        self.x_train = tf.transpose(self.x_train, [0, 3, 1, 2])
        self.x_valid = tf.transpose(self.x_valid, [0, 3, 1, 2])
        self.x_valid_rl = tf.transpose(self.x_valid_rl, [0, 3, 1, 2])
        self.x_test = tf.transpose(self.x_test, [0, 3, 1, 2])




  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None
    global_step = sess.run(self.global_step)
    print "Eval at {}".format(global_step)

    if eval_set == "train":
      assert self.x_train is not None
      assert self.train_acc is not None
      num_examples = self.num_train_examples
      num_batches = self.num_train_batches
      acc_op = self.train_acc
    elif eval_set == "valid":
      assert self.x_valid is not None
      assert self.valid_acc is not None
      num_examples = self.num_valid_examples
      num_batches = self.num_valid_batches
      acc_op = self.valid_acc
    elif eval_set == "test":
      assert self.test_acc is not None
      num_examples = self.num_test_examples
      num_batches = self.num_test_batches
      acc_op = self.test_acc
    else:
      raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

    total_acc = 0
    total_exp = 0
    for batch_id in xrange(num_batches):
      acc = sess.run(acc_op, feed_dict=feed_dict)
      total_acc += acc
      total_exp += self.eval_batch_size
      if verbose:
        sys.stdout.write("\r{:<5d}/{:>5d}".format(total_acc, total_exp))
    if verbose:
      print ""
    print "{}_accuracy: {:<6.4f}".format(
      eval_set, float(total_acc) / total_exp)

  def _build_train(self):
    print "Build train graph"
    logits = self._model(self.x_train, True)
    log_probs = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "-" * 80
    for var in tf_variables:
      print var

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

  def _build_valid(self):
    if self.x_valid is not None:
      print "-" * 80
      print "Build valid graph"
      logits = self._model(self.x_valid, False, reuse=True)
      self.valid_preds = tf.argmax(logits, axis=1)
      self.valid_preds = tf.to_int32(self.valid_preds)
      self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
      self.valid_acc = tf.to_int32(self.valid_acc)
      self.valid_acc = tf.reduce_sum(self.valid_acc)

  def _build_test(self):
    print "-" * 80
    print "Build test graph"
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  def build_valid_rl(self, shuffle=False):
    print "-" * 80
    print "Build valid graph on shuffled data"
    logits = self._model(self.x_valid_rl, False, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, self.y_valid_rl)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

  def _model(self, images, is_training, reuse=None):
    raise NotImplementedError("Abstract method")
