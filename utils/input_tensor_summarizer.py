from __future__ import absolute_import, division, print_function
import tensorflow as tf
from utils.learning import *

class InputTensorSummarizer(object):

    def __init__(self, features, labels, mode, hps):
        self.hps = hps
        TD = {}    # tensor_dict
        self.car1 = car1 = features["car1"]
        self.car2 = car2 = features["car2"]
        self.extras = extras = features["extras"]
        self.traj_lengths = features["traj_lengths"]

        with tf.variable_scope("data_rearranging"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                mhl, ph = self.hps.minimum_history_length, self.hps.prediction_horizon
                self.bag_idx = bag_idx = features["bag_idx"]
                self.prediction_timesteps = mhl - 1 + tf.mod(tf.random_uniform(self.traj_lengths.shape,
                                                                               maxval=2**31-1,
                                                                               dtype=tf.int32),
                                                             self.traj_lengths-mhl-ph+1)
                TD["car1_present"] = extract_subtensor_per_batch_element(car1, self.prediction_timesteps)     # [bs, state_dim]
                TD["car2_present"] = extract_subtensor_per_batch_element(car2, self.prediction_timesteps)     # [bs, state_dim]
                TD["extras_present"] = extract_subtensor_per_batch_element(extras, self.prediction_timesteps) # [bs, extras_dim]
                TD["bag_idx"] = extract_subtensor_per_batch_element(bag_idx, self.prediction_timesteps)       # [bs, 1]
                TD["car1_future"] = tf.stack([extract_subtensor_per_batch_element(car1, self.prediction_timesteps+i+1)
                                              for i in range(self.hps.prediction_horizon)], axis=1)           # [bs, ph, state_dim]
                TD["car2_future"] = tf.stack([extract_subtensor_per_batch_element(labels, self.prediction_timesteps+i+1)
                                              for i in range(self.hps.prediction_horizon)], axis=1)           # [bs, ph, state_dim]
            elif mode == tf.estimator.ModeKeys.EVAL:
                self.bag_idx = bag_idx = features["bag_idx"]
                TD["car1_present"] = self.extract_ragged_subarray(car1)                                  # [nbs, state_dim]
                TD["car2_present"] = self.extract_ragged_subarray(car2)                                  # [nbs, state_dim]
                TD["extras_present"] = self.extract_ragged_subarray(extras)                              # [nbs, extras_dim]
                TD["bag_idx"] = self.extract_ragged_subarray(bag_idx)                                    # [nbs, 1]
                TD["car1_future"] = tf.stack([self.extract_ragged_subarray(car1, i+1)
                                              for i in range(self.hps.prediction_horizon)], axis=1)      # [nbs, ph, state_dim]
                TD["car2_future"] = tf.stack([self.extract_ragged_subarray(labels, i+1)
                                              for i in range(self.hps.prediction_horizon)], axis=1)      # [nbs, ph, state_dim]
            elif mode == tf.estimator.ModeKeys.PREDICT:
                TD["car1_present"] = self.extract_subarray_ends(car1)                                    # [bs, state_dim]
                TD["car2_present"] = self.extract_subarray_ends(car2)                                    # [bs, state_dim]
                TD["extras_present"] = self.extract_subarray_ends(extras)                                # [bs, extras_dim]
                TD["car1_future"] = features["car1_future"]                                              # [bs, ph, state_dim]

            car2_prediction_present = tf.concat([TD["car2_present"][:,p:p+1]
                                                 for p in hps.pred_indices], axis=1)                     # [bs/nbs, pred_dim]
            TD["joint_present"] = tf.concat([TD["car1_present"], car2_prediction_present], axis=1)       # [bs/nbs, state_dim+pred_dim]

        TD["history_summaries"] = self.summarize_car_histories(mode)                                     # [bs/nbs, enc_rnn_dim]
        if mode == tf.estimator.ModeKeys.PREDICT: # and self.car1.shape[0] == 1: (this needs to be a tf.shape and a tf.cond)
            TD["joint_present"] = tf.tile(TD["joint_present"], [tf.shape(features["car1_future"])[0], 1])
            TD["car1_present"] = tf.tile(TD["car1_present"], [tf.shape(features["car1_future"])[0], 1])
            TD["history_summaries"] = tf.tile(TD["history_summaries"], [tf.shape(features["car1_future"])[0], 1])

        TD["car1_future_summaries"] = self.summarize_car_futures(TD["car1_present"], TD["car1_future"], mode, "car1")     # [bs/nbs, 2*enc_rnn_dim]
        TD["x"] = tf.concat([TD["history_summaries"], TD["car1_future_summaries"]], axis=1)              # [bs/nbs, 3*enc_rnn_dim]
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            TD["car2_future_summaries"] = self.summarize_car_futures(TD["car2_present"], TD["car2_future"], mode, "car2") # [bs/nbs, 2*enc_rnn_dim]
        self.tensor_dict = TD

    def summarize_car_histories(self, mode):
        with tf.variable_scope("car_history_rnn"):
            cell = stacked_rnn_cell(self.hps.rnn_cell,
                                    self.hps.rnn_cell_kwargs,
                                    self.hps.enc_rnn_dim_history,
                                    self.hps.rnn_io_dropout_keep_prob,
                                    mode)
            joint_history = tf.concat([self.car1, self.car2, self.extras], 2, name="joint_history")
            outputs, _ = tf.nn.dynamic_rnn(cell, joint_history, self.traj_lengths, dtype=tf.float32, time_major=False) # [bs, max_time, enc_rnn_dim]
            if mode == tf.estimator.ModeKeys.TRAIN:
                return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps)
            elif mode == tf.estimator.ModeKeys.EVAL:
                return self.extract_ragged_subarray(outputs)    # [nbs, enc_rnn_dim]
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return self.extract_subarray_ends(outputs)      # [bs, enc_rnn_dim]

    def summarize_car_futures(self, car_present, car_future, mode, scope):
        enc_rnn_dim = self.hps.enc_rnn_dim_future
        with tf.variable_scope("car_future_rnn"):
            cell = stacked_rnn_cell(self.hps.rnn_cell,
                                    self.hps.rnn_cell_kwargs,
                                    self.hps.enc_rnn_dim_future,
                                    self.hps.rnn_io_dropout_keep_prob,
                                    mode)
            with tf.variable_scope(scope):
                initial_state = project_to_RNN_initial_state(cell, car_present)
                outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, car_future,
                                                                 initial_state_fw=initial_state,
                                                                 dtype = tf.float32,
                                                                 time_major=False)
                return tf.concat([unpack_RNN_state(state[0]), unpack_RNN_state(state[1])], axis=1)

    def extract_ragged_subarray(self, tensor, offset=0):    # defines a new "batch size", call it "nbs"
        mhl, ph = self.hps.minimum_history_length, self.hps.prediction_horizon
        return tf.boolean_mask(tensor[:,mhl-1+offset:,:],
                               tf.sequence_mask(self.traj_lengths-mhl-ph+1, tensor.shape[1]-mhl+1-offset))

    def extract_subarray_ends(self, tensor, offset=0):      # ON THE CHOPPING BLOCK
        return extract_subtensor_per_batch_element(tensor, self.traj_lengths-1+offset)