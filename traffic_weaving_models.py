from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.distributions as distributions
import numpy as np

from utils.learning import *
from utils.input_tensor_summarizer import *

hps = tf.contrib.training.HParams(
    ### Training
    ## Batch Sizes
    batch_size = 32,
    ## Learning Rate
    learning_rate = 0.001,
    min_learning_rate = 0.00001,
    learning_decay_rate = 0.9999,
    ## Optimizer
    optimizer = tf.train.AdamOptimizer,
    optimizer_kwargs = {},
    grad_clip = 1.0,

    ### Prediction
    minimum_history_length = 5,    # 0.5 seconds
    prediction_horizon = 15,       # 1.5 seconds (at least as far as the loss function is concerned)

    ### Variational Objective
    ## Objective Formulation
    alpha = 1,
    k = 3,              # number of samples from z during training
    k_eval = 50,        # number of samples from z during evaluation
    use_iwae = False,   # only matters if alpha = 1
    kl_exact = True,    # relevant only if alpha = 1
    ## KL Annealing/Bounding
    kl_min = 0.07,
    kl_weight = 1.0,
    kl_weight_start = 0.0001,
    kl_decay_rate = 0.99995,
    kl_crossover = 8000,
    kl_sigmoid_divisor = 6,

    ### Network Parameters
    ## RNNs/Summarization
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell,
    rnn_cell_kwargs = {"layer_norm": False, "dropout_keep_prob": 0.75},
    MLP_dropout_keep_prob = 0.9,
    rnn_io_dropout_keep_prob = 1.0,
    enc_rnn_dim_history = [32],
    enc_rnn_dim_future = [32],
    dec_rnn_dim = [128],
    dec_GMM_proj_MLP_dims = None,
    sample_model_during_dec = True,
    dec_sample_model_prob_start = 0.0,
    dec_sample_model_prob_final = 0.0,
    dec_sample_model_prob_crossover = 20000,
    dec_sample_model_prob_divisor = 6,
    ## q_z_xy (encoder)
    q_z_xy_MLP_dims = None,
    ## p_z_x (encoder)
    p_z_x_MLP_dims = [16],
    ## p_y_xz (decoder)
    fuzz_factor = 0.05,
    GMM_components = 16,
    log_sigma_min = -10,
    log_sigma_max = 10,
    log_p_yt_xz_max = 50,

    ### Latent Variables
    latent_type = "discrete",
    ## Discrete Latent
    N = 2,
    K = 5,
    # Relaxed One-Hot Temperature Annealing
    tau_init = 2.0,
    tau_final = 0.001,
    tau_decay_rate = 0.9999,
    # Logit Clipping
    use_z_logit_clipping = False,
    z_logit_clip_start = 0.05,
    z_logit_clip_final = 3.0,
    z_logit_clip_crossover = 8000,
    z_logit_clip_divisor = 6,
    ## MVG Latent
    MVG_latent_dim = 32
)


class TrafficWeavingModel(object):

    def __init__(self):
        pass

    def setup_model(self, features, labels, mode, hps):
        pass

    def set_annealing_params(self):
        self.logging = {}
        with tf.variable_scope("batchwise_annealing"):
            self.lr = exp_anneal(self.hps.learning_rate, self.hps.min_learning_rate, self.hps.learning_decay_rate)
            self.logging["lr"] = self.lr
            tf.summary.scalar("lr", self.lr)

    def train_loss(self, tensor_dict):
        raise Exception("loss function must be overridden by child class of TrafficWeavingModel")

    def eval_loss(self, tensor_dict):
        return self.train_loss(tensor_dict)

    def standardize_features(self, features, mode):
        # features: {car1, car2, traj_lengths}
        # feature_standardization: subdictionary of hps
        with tf.variable_scope("features_standardization"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                cars_standardization = self.hps.cars_standardization
                m = tf.Variable(cars_standardization["mean"], trainable=False, name="train_mean")
                s = tf.Variable(cars_standardization["std"], trainable=False, name="train_std")
                extras_standardization = self.hps.extras_standardization
                em = tf.Variable(extras_standardization["mean"], trainable=False, name="train_extras_mean")
                es = tf.Variable(extras_standardization["std"], trainable=False, name="train_extras_std")
            else:    # these variables will get "restore"d over during EVAL/PREDICT
                zero_state = np.zeros(self.state_dim, dtype=np.float32)
                m = tf.Variable(zero_state, trainable=False, name="train_mean")
                s = tf.Variable(zero_state, trainable=False, name="train_std")
                zero_extras = np.zeros(self.extras_dim, dtype=np.float32)
                em = tf.Variable(zero_extras, trainable=False, name="train_extras_mean")
                es = tf.Variable(zero_extras, trainable=False, name="train_extras_std")

            def standardize_feature(key, mean, std):
                with tf.variable_scope(key):
                    std_data = standardize(tf.to_float(features[key]), mean, std)
                    if mode == tf.estimator.ModeKeys.TRAIN and self.hps.fuzz_factor > 0:
                        return std_data + self.hps.fuzz_factor*tf.random_normal(std_data.shape)
                    return std_data

            features_standardized = {
                "car1": standardize_feature("car1", m, s),               # [batch_size, max_time, state_dim]
                "car2": standardize_feature("car2", m, s),               # [batch_size, max_time, state_dim]
                "extras": standardize_feature("extras", em, es),         # [batch_size, max_time, extras_dim]
                "traj_lengths": tf.to_int32(features["traj_lengths"]),   # [batch_size]
            }
            if "bag_idx" in features:    # True at TRAIN, EVAL time
                features_standardized["bag_idx"] = tf.to_int32(features["bag_idx"])    # [batch_size, max_time, 1]

            if mode == tf.estimator.ModeKeys.PREDICT:
                features_standardized["car1_future"] = standardize_feature("car1_future", m, s)

        return features_standardized

    def standardize_labels(self, labels, mode):
        with tf.variable_scope("label_standardization"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                label_standardization = self.hps.label_standardization
                m = tf.Variable(label_standardization["mean"], trainable=False, name="train_mean")
                s = tf.Variable(label_standardization["std"], trainable=False, name="train_std")
            else:    # these variables will get "restore"d over during EVAL/PREDICT
                zero_state = np.zeros(self.pred_dim, dtype=np.float32)
                m = tf.Variable(zero_state, trainable=False, name="train_mean")
                s = tf.Variable(zero_state, trainable=False, name="train_std")

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                std_data = standardize(tf.to_float(labels), m, s)
                if mode == tf.estimator.ModeKeys.TRAIN and self.hps.fuzz_factor > 0:
                    return std_data + self.hps.fuzz_factor*tf.random_normal(std_data.shape)
                return std_data
            elif mode == tf.estimator.ModeKeys.PREDICT:
                self.labels_m = m
                self.labels_s = s
                return None

    def optimizer(self, loss):
        with tf.variable_scope("optimizer"):
            opt = self.hps.optimizer(learning_rate=self.lr, **self.hps.optimizer_kwargs)
            if self.hps.grad_clip is not None:
                gvs = opt.compute_gradients(self.loss)
                g = self.hps.grad_clip
                clipped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
                train_op = opt.apply_gradients(clipped_gvs, global_step=tf.train.get_global_step())
            else:
                train_op = opt.minimize(loss, tf.train.get_global_step())
        return train_op

    def model_fn(self, features, labels, mode, params):
        self.setup_model(features, labels, mode, params)

        self.mode = mode
        self.hps = params
        self.predictions_dict = None
        self.eval_metric_ops = None
        self.loss = None
        self.train_op = None
        self.temp = None
        self.lr = None
        self.kl_weight = None
        self.logging = {}

        # standardize the features, returns a dictionary of {car1, car2, traj_lengths}
        features_standardized = self.standardize_features(features, mode)
        # standardize the labels
        labels_standardized = self.standardize_labels(labels, mode)

        # prepares the features by using RNNs to summarize
        tensor_dict = InputTensorSummarizer(features_standardized, labels_standardized, mode, params).tensor_dict

        if mode == tf.estimator.ModeKeys.TRAIN:
            # annealing function <- inputs into other functions?
            self.set_annealing_params()
            # get loss function <- where all the subclasses get real
            self.loss = self.train_loss(tensor_dict)
            # optimizer
            self.train_op = self.optimizer(self.loss)
        elif mode == tf.estimator.ModeKeys.EVAL:
            self.loss = self.eval_loss(tensor_dict)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            self.predictions_dict = self.make_predictions_dict(tensor_dict)

        export_outputs = {"predictions": tf.estimator.export.PredictOutput(self.predictions_dict)} if self.predictions_dict else None
        return tf.estimator.EstimatorSpec(mode, self.predictions_dict, self.loss, self.train_op,
                                          training_hooks=[tf.train.LoggingTensorHook(self.logging, every_n_iter=100)],
                                          eval_metric_ops=self.eval_metric_ops,
                                          export_outputs=export_outputs)


class TrafficWeavingCVAE(TrafficWeavingModel):

    def setup_model(self, features, labels, mode, hps):
        if "car1_future_x" in features and "car1_future_y" in features:
            features["car1_future"] = cartesian_product_over_batch(features["car1_future_x"],
                                                                   features["car1_future_y"],
                                                                   name="car1_future")
            features.pop("car1_future_x")
            features.pop("car1_future_y")
        self.pred_dim = len(hps.pred_indices)    # labels.shape[-1].value
        self.state_dim = features["car1"].shape[-1].value
        self.extras_dim = features["extras"].shape[-1].value
        if hps.latent_type == "MVG":
            self.latent = MVGLatent(hps.MVG_latent_dim)
        elif hps.latent_type == "discrete":
            # N = number of variables, K = categories per variable
            self.latent = DiscreteLatent(hps.N, hps.K)
            self.latent.kl_min = hps.kl_min

        with tf.variable_scope("sample_ct"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.sample_ct = hps.k
            elif mode == tf.estimator.ModeKeys.EVAL:
                self.sample_ct = hps.k_eval
            elif mode == tf.estimator.ModeKeys.PREDICT:
                self.sample_ct = features["sample_ct"][0]

    def set_annealing_params(self):
        super(TrafficWeavingCVAE, self).set_annealing_params()
        with tf.variable_scope("batchwise_annealing", reuse=True):
            if np.abs(self.hps.alpha - 1.0) < 1e-3 and not self.hps.use_iwae:
                self.kl_weight = sigmoid_anneal(self.hps.kl_weight_start, self.hps.kl_weight,
                                                self.hps.kl_crossover, self.hps.kl_crossover / self.hps.kl_sigmoid_divisor)
                self.logging["kl_weight"] = self.kl_weight
                tf.summary.scalar("kl_weight", self.kl_weight)

            if self.hps.sample_model_during_dec:
                self.dec_sample_model_prob = sigmoid_anneal(
                    self.hps.dec_sample_model_prob_start,
                    self.hps.dec_sample_model_prob_final,
                    self.hps.dec_sample_model_prob_crossover,
                    self.hps.dec_sample_model_prob_crossover / self.hps.dec_sample_model_prob_divisor
                )
                tf.summary.scalar("dec_sample_model_prob", self.dec_sample_model_prob)

            if self.hps.latent_type == "discrete":
                self.latent.temp = exp_anneal(self.hps.tau_init, self.hps.tau_final, self.hps.tau_decay_rate)
                self.logging["temp"] = self.latent.temp
                tf.summary.scalar("temp", self.latent.temp)
                if self.hps.use_z_logit_clipping:
                    self.latent.z_logit_clip = sigmoid_anneal(self.hps.z_logit_clip_start, self.hps.z_logit_clip_final,
                                                              self.hps.z_logit_clip_crossover,
                                                              self.hps.z_logit_clip_crossover / self.hps.z_logit_clip_divisor)
                    tf.summary.scalar("z_logit_clip", self.z_logit_clip)

    def q_z_xy(self, x, y, mode):
        with tf.variable_scope("q_z_xy"):
            xy = tf.concat([x, y], 1)
            # h = xy    # https://arxiv.org/pdf/1703.10960.pdf, https://arxiv.org/pdf/1704.03477.pdf
            h = MLP(xy, self.hps.q_z_xy_MLP_dims, tf.nn.relu, self.hps.MLP_dropout_keep_prob, mode)
            return self.latent.dist_from_h(h, mode)

    def p_z_x(self, x, mode):
        with tf.variable_scope("p_z_x"):
            # h = tf.layers.dense(x, h_dim, activation=tf.nn.relu, name="dense")    # https://arxiv.org/pdf/1703.10960.pdf
            h = MLP(x, self.hps.p_z_x_MLP_dims, tf.nn.relu, self.hps.MLP_dropout_keep_prob, mode)
            return self.latent.dist_from_h(h, mode)

    def p_y_xz(self, x, z_stacked, TD, mode):
        # x is [bs/nbs, 2*enc_rnn_dim]
        # z_stacked is [k, bs/nbs, N*K]    (at EVAL or PREDICT time, k (=self.sample_ct) may be hps.k, K**N or sample_ct)
        # in this function, rnn decoder inputs are of the form: z + x + car1 + car2 (note: first 3 are "extras" to help with learning)
        ph = self.hps.prediction_horizon

        k, GMM_c, pred_dim = self.sample_ct, self.hps.GMM_components, self.pred_dim
        with tf.variable_scope("p_y_xz") as varscope:
            z = tf.reshape(z_stacked, [-1, self.latent.z_dim])               # [k;bs/nbs, z_dim]
            zx = tf.concat([z, tf.tile(x, [k, 1])], axis=1)           # [k;bs/nbs, z_dim + 2*enc_rnn_dim]

            cell = stacked_rnn_cell(self.hps.rnn_cell,
                                    self.hps.rnn_cell_kwargs,
                                    self.hps.dec_rnn_dim,
                                    self.hps.rnn_io_dropout_keep_prob,
                                    mode)
            initial_state = project_to_RNN_initial_state(cell, zx)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if self.hps.sample_model_during_dec and mode == tf.estimator.ModeKeys.TRAIN:
                    input_ = tf.concat([zx, tf.tile(TD["joint_present"], [k, 1])], axis=1)    # [k;bs, N*K + 2*enc_rnn_dim + pred_dim+state_dim]
                    state = initial_state
                    with tf.variable_scope("rnn") as rnnscope:
                        log_pis, mus, log_sigmas, corrs = [], [], [], []
                        for j in range(ph):
                            if j > 0:
                                rnnscope.reuse_variables()
                            output, state = cell(input_, state)
                            log_pi_t, mu_t, log_sigma_t, corr_t = project_to_GMM_params(output, GMM_c, pred_dim, self.hps.dec_GMM_proj_MLP_dims) 
                            y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                                        self.hps.log_sigma_min, self.hps.log_sigma_max).sample()              # [k;bs, pred_dim]
                            mask = distributions.Bernoulli(probs=self.dec_sample_model_prob,
                                                           dtype=tf.float32).sample((tf.shape(y_t)[0], 1))    # maybe tf.shape
                            y_t = mask*y_t + (1 - mask)*tf.tile(TD["car2_future"][:,j,:], [k, 1])
                            log_pis.append(log_pi_t)
                            mus.append(mu_t)
                            log_sigmas.append(log_sigma_t)
                            corrs.append(corr_t)
                            car_inputs = tf.concat([tf.tile(TD["car1_future"][:,j,:], [k, 1]), y_t], axis=1)  # [k;bs, state_dim + pred_dim]
                            input_ = tf.concat([zx, car_inputs], axis=1)                # [k;bs, N*K + 2*enc_rnn_dim + state_dim + pred_dim]  
                        log_pis = tf.stack(log_pis, axis=1)                             # [k;bs, ph, GMM_c]
                        mus = tf.stack(mus, axis=1)                                     # [k;bs, ph, GMM_c*pred_dim]
                        log_sigmas = tf.stack(log_sigmas, axis=1)                       # [k;bs, ph, GMM_c*pred_dim]
                        corrs = tf.stack(corrs, axis=1)                                 # [k;bs, ph, GMM_c]
                else:
                    zx_with_time_dim = tf.expand_dims(zx, 1)                           # [k;bs/nbs, 1, N*K + 2*enc_rnn_dim]
                    zx_time_tiled = tf.tile(zx_with_time_dim, [1, ph, 1])              # [k;bs/nbs, ph, N*K + 2*enc_rnn_dim]
                    car_inputs = tf.concat([                                           # [bs/nbs, ph, 2*state_dim]
                        tf.expand_dims(TD["joint_present"], 1),                                          # [bs/nbs, 1, state_dim+pred_dim]
                        tf.concat([TD["car1_future"][:,:ph-1,:], TD["car2_future"][:,:ph-1,:]], axis=2)  # [bs/nbs, ph-1, state_dim+pred_dim]
                        ], axis=1)
                    inputs = tf.concat([zx_time_tiled, tf.tile(car_inputs, [k, 1, 1])], axis=2)  # [k;bs/nbs, ph, N*K + 2*enc_rnn_dim + pred_dim + state_dim]
                    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state,    # [k;bs/nbs, ph, dec_rnn_dim]
                                                                 time_major=False,
                                                                 dtype=tf.float32,
                                                                 scope="rnn")
                    with tf.variable_scope("rnn"):    # required to match PREDICT mode below
                        log_pis, mus, log_sigmas, corrs = project_to_GMM_params(outputs, GMM_c, pred_dim, self.hps.dec_GMM_proj_MLP_dims)

                tf.summary.histogram("GMM_log_pis", log_pis)
                tf.summary.histogram("GMM_log_sigmas", log_sigmas)
                tf.summary.histogram("GMM_corrs", corrs)

            elif mode == tf.estimator.ModeKeys.PREDICT:
                input_ = tf.concat([zx, tf.tile(TD["joint_present"], [k, 1])], axis=1)    # [k;bs, N*K + 2*enc_rnn_dim + pred_dim+state_dim]
                state = initial_state
                with tf.variable_scope("rnn") as rnnscope:
                    log_pis, mus, log_sigmas, corrs, y = [], [], [], [], []
                    for j in range(ph):
                        if j > 0:
                            rnnscope.reuse_variables()
                        output, state = cell(input_, state)
                        log_pi_t, mu_t, log_sigma_t, corr_t = project_to_GMM_params(output, GMM_c, pred_dim, self.hps.dec_GMM_proj_MLP_dims) 
                        y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                                    self.hps.log_sigma_min, self.hps.log_sigma_max).sample()              # [k;bs, pred_dim]
                        log_pis.append(log_pi_t)
                        mus.append(mu_t)
                        log_sigmas.append(log_sigma_t)
                        corrs.append(corr_t)
                        y.append(y_t)
                        car_inputs = tf.concat([tf.tile(TD["car1_future"][:,j,:], [k, 1]), y_t], axis=1)  # [k;bs, state_dim + pred_dim]
                        input_ = tf.concat([zx, car_inputs], axis=1)                # [k;bs, N*K + 2*enc_rnn_dim + state_dim + pred_dim]  
                    log_pis = tf.stack(log_pis, axis=1)                             # [k;bs, ph, GMM_c]
                    mus = tf.stack(mus, axis=1)                                     # [k;bs, ph, GMM_c*pred_dim]
                    log_sigmas = tf.stack(log_sigmas, axis=1)                       # [k;bs, ph, GMM_c*pred_dim]
                    corrs = tf.stack(corrs, axis=1)                                 # [k;bs, ph, GMM_c]
                    car2_sampled_future = tf.reshape(tf.stack(y, axis=1), [k, -1, ph, pred_dim])  # [k, bs, ph, pred_dim]

            y_dist = GMM2D(tf.reshape(log_pis, [k, -1, ph, GMM_c]),
                           tf.reshape(mus, [k, -1, ph, GMM_c*pred_dim]),
                           tf.reshape(log_sigmas, [k, -1, ph, GMM_c*pred_dim]),
                           tf.reshape(corrs, [k, -1, ph, GMM_c]),
                           self.hps.log_sigma_min,
                           self.hps.log_sigma_max)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return y_dist, car2_sampled_future
            else:
                return y_dist

    def encoder(self, x, y, mode, reuse=None):
        k = self.sample_ct
        with tf.variable_scope("encoder", reuse=reuse):
            self.latent.q_dist = self.q_z_xy(x, y, mode)
            self.latent.p_dist = self.p_z_x(x, mode)
            z = self.latent.sample_q(k, mode)
            if mode == tf.estimator.ModeKeys.TRAIN and self.hps.kl_exact:
                kl_obj = self.latent.kl_q_p()
                tf.summary.scalar("kl", kl_obj)
            else:
                kl_obj = None

            return z, kl_obj

    def decoder(self, x, y, z, TD, mode, reuse=None):
        # x is [nbs, 2*enc_rnn_dim]
        # y is [nbs, prediction_horizon, pred_dim]
        # z is [k, bs/nbs, N*K]
        ph = self.hps.prediction_horizon
        pred_dim = self.pred_dim

        with tf.variable_scope("decoder", reuse=reuse):
            self.y_dist = y_dist = self.p_y_xz(x, z, TD, mode)
            if len(y_dist.batch_shape) == 2:    # e.g., y_dist is a MVG with loc (mean) shape [k, nbs, ph*pred_dim]
                y_vector = tf.reshape(y, [-1, ph*pred_dim])               # [nbs, ph*pred_dim]
                log_p_y_xz = y_dist.log_prob(y_vector)                    # [k, nbs]
            elif len(y_dist.batch_shape) == 3:  # e.g., y_dist is a GMM with mus [k, nbs, ph, GMM_c;pred_dim]
                log_p_yt_xz = tf.minimum(y_dist.log_prob(y), self.hps.log_p_yt_xz_max)
                log_p_y_xz = tf.reduce_sum(log_p_yt_xz, axis=2)    # [k, nbs]
                tf.summary.histogram("log_p_yt_xz", log_p_yt_xz)
            return log_p_y_xz

    def train_loss(self, TD):
        mode = tf.estimator.ModeKeys.TRAIN
        z, kl = self.encoder(TD["x"], TD["car2_future_summaries"], mode)        # [k, nbs, N*K], [] (i.e., scalar)
        log_p_y_xz = self.decoder(TD["x"], TD["car2_future"], z, TD, mode)          # [k, nbs]

        if np.abs(self.hps.alpha - 1.0) < 1e-3 and not self.hps.use_iwae:
            log_p_y_xz_mean = tf.reduce_mean(log_p_y_xz, 0)                       # [nbs]
            tf.summary.histogram("log_p_y_xz", log_p_y_xz_mean)
            log_likelihood = tf.reduce_mean(log_p_y_xz_mean)
            ELBO = log_likelihood - self.kl_weight*kl
            loss = -ELBO
        else:
            log_q_z_xy = self.latent.q_log_prob(z)    # [k, nbs]
            log_p_z_x = self.latent.p_log_prob(z)     # [k, nbs]
            a = self.hps.alpha
            log_pp_over_q = log_p_y_xz + log_p_z_x - log_q_z_xy
            log_likelihood = (tf.reduce_mean(tf.reduce_logsumexp(log_pp_over_q*(1-a), axis=0)) -\
                                   tf.log(tf.to_float(self.hps.k))) / (1-a)
            loss = -log_likelihood

        tf.summary.scalar("log_likelihood", log_likelihood)
        tf.summary.scalar("loss", loss)
        self.latent.summarize_for_tensorboard()
        return loss

    def eval_loss(self, TD):
        mode = tf.estimator.ModeKeys.EVAL

        ### Importance sampled NLL estimate
        z, _ = self.encoder(TD["x"], TD["car2_future_summaries"], mode)      # [k_eval, nbs, N*K]
        log_p_y_xz = self.decoder(TD["x"], TD["car2_future"], z, TD, mode)       # [k_eval, nbs]
        log_q_z_xy = self.latent.q_log_prob(z)                          # [k_eval, nbs]
        log_p_z_x = self.latent.p_log_prob(z)                           # [k_eval, nbs]
        log_likelihood = tf.reduce_mean(tf.reduce_logsumexp(log_p_y_xz + log_p_z_x - log_q_z_xy, axis=0)) -\
                         tf.log(tf.to_float(self.sample_ct))
        self.eval_metric_ops = {"NLL_q (IS)": tf.metrics.mean(-log_likelihood)}
        loss = -log_likelihood

        ### Naive sampled NLL estimate
        z = self.latent.sample_p(self.sample_ct, mode)
        log_p_y_xz = self.decoder(TD["x"], TD["car2_future"], z, TD, mode, reuse=True)
        log_likelihood_p = tf.reduce_mean(tf.reduce_logsumexp(log_p_y_xz, axis=0)) - tf.log(tf.to_float(self.sample_ct))
        self.eval_metric_ops["NLL_p"] = tf.metrics.mean(-log_likelihood_p)

        ### Exact NLL
        K, N = self.hps.K, self.hps.N
        if self.hps.latent_type == "discrete" and K**N < 50:
            self.sample_ct = K ** N
            nbs = tf.shape(TD["x"])[0]
            z_raw = tf.tile(all_one_hot_combinations(N, K, np.float32), multiples=[1, nbs])    # [K**N, nbs*N*K]
            z = tf.reshape(z_raw, [K**N, -1, N*K])                                                  # [K**N, nbs, N*K]
            log_p_y_xz = self.decoder(TD["x"], TD["car2_future"], z, TD, mode, reuse=True)                    # [K**N, nbs]
            log_p_z_x = self.latent.p_log_prob(z)                                              # [K**N, nbs]
            exact_log_likelihood = tf.reduce_mean(tf.reduce_logsumexp(log_p_y_xz + log_p_z_x, axis=0))
            self.eval_metric_ops["NLL_exact"] = tf.metrics.mean(-exact_log_likelihood)

        return loss

    def make_predictions_dict(self, TD):
        mode = tf.estimator.ModeKeys.PREDICT

        with tf.variable_scope("encoder"):
            self.latent.p_dist = self.p_z_x(TD["x"], mode)

        z = self.latent.sample_p(self.sample_ct, mode)

        with tf.variable_scope("decoder"):
            y_dist, car2_sampled_future = self.p_y_xz(TD["x"], z, TD, mode)      # y_dist.mean is [k, bs, ph*state_dim]

        with tf.variable_scope("outputs"):
            y = unstandardize(car2_sampled_future, self.labels_m, self.labels_s)

            predictions_dict = {"y": y, "z": z}
            predictions_dict = {k: tf.identity(v, name=k) for k,v in predictions_dict.items()}
            return predictions_dict


# class TrafficWeavingCVAE_ExtraInfo(TrafficWeavingCVAE):

#     def setup_model(self, features, labels, mode, hps):
#         self.pred_dim = len(hps.pred_indices) #labels.shape[-1].value
#         self.state_dim = features["car1"].shape[-1].value
#         self.extras_dim = features["extras"].shape[-1].value
#         if hps.latent_type == "MVG":
#             self.latent = MVGLatent(hps.MVG_latent_dim)
#         elif hps.latent_type == "discrete":
#             # N = number of variables, K = categories per variable
#             self.latent = DiscreteLatent(hps.N, hps.K)
#             self.latent.kl_min = hps.kl_min

#         with tf.variable_scope("sample_ct"):
#             if mode == tf.estimator.ModeKeys.TRAIN:
#                 self.sample_ct = hps.k
#             elif mode == tf.estimator.ModeKeys.EVAL:
#                 self.sample_ct = hps.k_eval
#             elif mode == tf.estimator.ModeKeys.PREDICT:
#                 self.input_sample_ct = features["sample_ct"][0]
#                 if hps.latent_type != "discrete" or hps.K ** hps.N > 100:
#                     self.sample_ct = self.input_sample_ct
#                 else:
#                     self.sample_ct = tf.cond(tf.equal(self.input_sample_ct, 0),
#                                              lambda: hps.K ** hps.N,
#                                              lambda: self.input_sample_ct)

#     def make_predictions_dict(self, TD):
#         mode = tf.estimator.ModeKeys.PREDICT

#         with tf.variable_scope("encoder"):
#             self.latent.p_dist = self.p_z_x(TD["x"], mode)

#         z = self.latent.sample_p(self.input_sample_ct, mode)    # [k, bs, N*K]

#         with tf.variable_scope("decoder"):
#             y_dist, car2_sampled_future = self.p_y_xz(TD["x"], z, TD, mode)    # y_dist.mean is [k, bs, ph*state_dim]

#         with tf.variable_scope("outputs"):
#             if type(y_dist) is distributions.MultivariateNormalDiag:    # really should be anything with defined mean, cov
#                 mu = unstandardize(y_dist.mean(), self.labels_m, self.labels_s)
#                 Sigma = unstandardize(tf.transpose(
#                             unstandardize(tf.transpose(y_dist.covariance(),
#                                                        [0,1,3,2]), self.labels_m, self.labels_s, False),
#                                                    [0,1,3,2]), self.labels_m, self.labels_s, False)
#             else:
#                 mu = tf.zeros([1])
#                 Sigma = tf.zeros([1])
#             y = unstandardize(car2_sampled_future, self.labels_m, self.labels_s)
#             p = self.latent.get_p_dist_params()
#             p_z = tf.exp(self.latent.p_log_prob(z))

#             predictions_dict = {
#                                     "mu": mu,
#                                     "Sigma": Sigma,
#                                     "y": y,
#                                     "z": z,
#                                     "p": p,
#                                     "p_z": p_z
#                                 }
#             predictions_dict = {k: tf.identity(v, name=k) for k,v in predictions_dict.items()}
#             return predictions_dict
