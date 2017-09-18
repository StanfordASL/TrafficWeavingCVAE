#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import rospy
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from traffic_weaving_prediction.msg import prediction_input, prediction_output
from numpy_ros import numpy_to_multiarray, multiarray_to_numpy
dir(tf.contrib)

# model_dir = "/data/laneswap/exported_models/1505240770"
# model_dir = "/data/laneswap/exported_models/1505503714"     # one gmm component
# model_dir = "/data/laneswap/exported_models/1505507954"     # 1 latent dimension
model_dir = rospy.get_param("model", "") or "/data/trafficweaving/models/1507924360"

class Predictor(object):

    def __init__(self):
        with tf.Graph().as_default() as g:
            self.sess = tf.Session()
            print("Loading model from: " + model_dir)
            tf.saved_model.loader.load(self.sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       model_dir)

        rospy.init_node("cvae_prediction", anonymous=True)
        rospy.Subscriber("prediction_input", prediction_input, self.prediction_callback)
        self.pub = rospy.Publisher("prediction_output", prediction_output, queue_size=10)

    def prediction_callback(self, input_msg):
        tic = timeit.default_timer()
        print("subscribed to prediction input")
        tic0 = timeit.default_timer()
        feed_dict = {"car1:0": multiarray_to_numpy(input_msg.car1),
                     "car2:0": multiarray_to_numpy(input_msg.car2),
                     "extras:0": multiarray_to_numpy(input_msg.extras),
                     "traj_lengths:0": multiarray_to_numpy(input_msg.traj_lengths),
                     "car1_future:0": multiarray_to_numpy(input_msg.car1_future),
                     "sample_ct:0": [input_msg.sample_ct]}
        toc0 = timeit.default_timer()

        print("constructing feed_dict took: ", toc0 - tic0, " (s), running tf!")

        tic0 = timeit.default_timer()
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        y, z = self.sess.run(["outputs/y:0",
                              "outputs/z:0"], 
                              feed_dict=feed_dict)
                              # options=options,
                              # run_metadata=run_metadata)
        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('/home/schmrlng/timeline' +
        #           os.environ["CUDA_VISIBLE_DEVICES"] +
        #           '.json', 'w') as f:
        #     f.write(chrome_trace)
        toc0 = timeit.default_timer()

        print("done running tf!, took (s): ", toc0 - tic0)

        tic0 = timeit.default_timer()
        output_msg = prediction_output()
        output_msg.y = numpy_to_multiarray(y)
        output_msg.z = numpy_to_multiarray(z)
        self.pub.publish(output_msg)
        toc0 = timeit.default_timer()
        toc = timeit.default_timer()

        print("output_msg constructed and published, took (s): ", toc0 - tic0)
        print("total time taken (s): ", toc - tic)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    p = Predictor()
    p.run()
