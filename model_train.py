import os
import warnings
# Dependency imports
from absl import flags
import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import tensorflow_hub as hub
tfd = tfp.distributions
from scipy.stats import norm
import matplotlib.pyplot as plt

import seaborn as sns

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# hide all warnings
import warnings
warnings.filterwarnings('ignore')


def USEembeddings(msg, myeb, insess):
    return insess.run(myeb([msg]))

def LoadPredictor(ifrestore=True):
    global sess

    sherpa_features = np.load('./datasets/features.npy')
    sherpa_labels = np.load('./datasets/labels.npy')
    sherpa_questions = pickle.load(open('./datasets/all_questions.pickle','rb'))

    mymodel = Model(sherpa_features, sherpa_labels)

    my_path = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(my_path, "bayesnn_tmp")

    if ifrestore==True:
        sess = mymodel.restore_model(save_path)
        return mymodel, mymodel.get_initializer(), mymodel.get_graph(), sess
    else:
        return mymodel, mymodel.get_initializer(), mymodel.get_graph()


class Model():

    def build_input_pipeline(self, train_features, train_labels, val_features, val_labels, batch_size, heldout_size):
        """Build an Iterator switching between train and heldout data."""
        # Build an iterator over training batches.
        training_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, np.int32(train_labels)))
        training_batches = training_dataset.shuffle(
            500, reshuffle_each_iteration=True).repeat().batch(batch_size)
        training_iterator = training_batches.make_one_shot_iterator()

        # Build a iterator over the heldout set with batch_size=heldout_size,
        # i.e., return the entire heldout set as a constant.
        heldout_dataset = tf.data.Dataset.from_tensor_slices(
            (val_features, np.int32(val_labels)))
        heldout_frozen = (heldout_dataset.take(heldout_size).
                          repeat().batch(heldout_size))
        heldout_iterator = heldout_frozen.make_one_shot_iterator()

        # Combine these into a feedable iterator that can switch between training
        # and validation inputs.
        handle = tf.placeholder(tf.string, shape=[])
        feedable_iterator = tf.data.Iterator.from_string_handle(
            handle, training_batches.output_types, training_batches.output_shapes)
        embeddings, labels = feedable_iterator.get_next()

        return embeddings, labels, handle, training_iterator, heldout_iterator

    def __init__(self, features, sherpa_data_labels):

        with tf.Graph().as_default() as self.grph:

            self.batch_size = 50
            self.data_dir = "./bayesnn_tmp/data/"
            self.model_dir = "./bayesnn_tmp/"
            self.num_monte_carlo = 1000
            self.learning_rate = 0.001

            self.features = features
            self.labels = sherpa_data_labels

            self.embeddings_train, self.labels_train, self.handle, self.training_iterator, self.heldout_iterator = self.build_input_pipeline(
                self.features,
                self.labels,
                self.features,
                self.labels,
                self.batch_size,
                len(features))

            # adding the iterator handle to meta-graph
            tf.add_to_collection('iterator_handle', self.handle)

            with tf.name_scope("bayesian_neural_net", values=[self.embeddings_train]):
                neural_net = tf.keras.Sequential([
                    tfp.layers.DenseLocalReparameterization(128, activation=tf.nn.relu),
                    tfp.layers.DenseLocalReparameterization(64, activation=tf.nn.relu),
                    tfp.layers.DenseLocalReparameterization(9)
                ])

                self.logits = neural_net(self.embeddings_train)
                self.labels_distribution = tfd.Categorical(logits=self.logits)

            # Compute the -ELBO as the loss, averaged over the batch size.
            self.neg_log_likelihood = -tf.reduce_mean(self.labels_distribution.log_prob(self.labels_train))
            self.kl = sum(neural_net.losses) / 600
            self.elbo_loss = self.neg_log_likelihood + self.kl

            # Build metrics for evaluation. Predictions are formed from a single forward
            # pass of the probabilistic layers. They are cheap but noisy predictions.
            self.predictions = tf.argmax(self.logits, axis=1)

            # Extract weight posterior statistics for layers with weight distributions
            # for later visualization.
            self.names = []
            self.qmeans = []
            self.qstds = []

            for i, layer in enumerate(neural_net.layers):
                try:
                    q = layer.kernel_posterior
                except AttributeError:
                    continue
                self.names.append("Layer {}".format(i))
                self.qmeans.append(q.mean())
                self.qstds.append(q.stddev())

            self.accuracy, self.accuracy_update_op = tf.metrics.accuracy(labels=self.labels_train,
                                                                         predictions=self.predictions)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.elbo_loss)

            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            # self.sess = tf.Session()
            # self.sess.run(self.init_op)

    def get_graph(self):
        return self.grph

    def get_initializer(self):
        return self.init_op

    def initialize_uninitialized_vars(self, sess):
        from itertools import compress
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                       for var in global_vars])
        not_initialized_vars = list(compress(global_vars, is_not_initialized))

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))


    def restore_model(self, save_path):
        global sess

        with self.grph.as_default() as loaded_graph:
            # print(">>> Variables in saved checkpoint <<<" + str(save_path))
            # vars_in_checkpoint = tf.train.list_variables(save_path)

            # for j in vars_in_checkpoint:
            #    print(j)

            sess = tf.Session(graph=loaded_graph)
            sess.run(self.init_op)
            new_saver = tf.train.Saver(save_relative_paths=True)
            new_saver.restore(sess, tf.train.latest_checkpoint(save_path))
            self.handle = tf.get_collection('iterator_handle')[0]

            return sess

    def update_model(self, epoch):

        my_path = os.path.abspath(os.path.dirname("__file__"))
        save_path = os.path.join(my_path, self.model_dir)
        mysession = self.restore_model(save_path)

        with self.grph.as_default():
            if tf.gfile.Exists(self.model_dir):
                tf.logging.warning("Warning: deleting old log directory at {}".format(self.model_dir))
                tf.gfile.DeleteRecursively(self.model_dir)

            tf.gfile.MakeDirs(self.model_dir)

            saver = tf.train.Saver(save_relative_paths=True)

            train_handle = mysession.run(self.training_iterator.string_handle())

            for step in range(epoch):
                _ = mysession.run([self.train_op, self.accuracy_update_op], feed_dict={self.handle: train_handle})
                if step % 100 == 0:
                    loss_value, accuracy_value = mysession.run([self.elbo_loss, self.accuracy],
                                                               feed_dict={self.handle: train_handle})
                    print(
                        "Step: {:>3d} Loss: {:.3f} Training Accuracy: {:.3f}".format(step, loss_value, accuracy_value))

            save_path = os.path.join(save_path, "bnn.epoch.{}.ckpt".format(epoch))
            saver.save(mysession, save_path)

    def predict_1(self, mysess, newdata_features):
        # we dont need the labels so we randomly generate some random array
        pseudo_labels = np.random.randint(3, size=len(newdata_features))

        with mysess as session:
            # newdata expects a numpy array
            heldout_dataset = tf.data.Dataset.from_tensor_slices((newdata_features, np.int32(pseudo_labels)))
            heldout_frozen = (heldout_dataset.take(len(newdata_features)).repeat().batch(len(newdata_features)))
            new_iterator = heldout_frozen.make_one_shot_iterator()
            new_handle = session.run(new_iterator.string_handle())
            probs = np.asarray(
                [mysess.run((self.labels_distribution.probs), feed_dict={self.handle: new_handle}) for _ in
                 range(self.num_monte_carlo)])

        return probs

    def predict_N(self, mysess, newdata_features):
        # get the size of newdata_features
        n_size = len(newdata_features)
        pseudo_labels = np.random.randint(3, size=n_size)
        with mysess as session:
            # newdata expects a numpy array
            heldout_dataset = tf.data.Dataset.from_tensor_slices((newdata_features, np.int32(pseudo_labels)))
            heldout_frozen = (heldout_dataset.take(len(newdata_features)).repeat().batch(len(newdata_features)))
            new_iterator = heldout_frozen.make_one_shot_iterator()
            new_handle = session.run(new_iterator.string_handle())
            probs = np.asarray(
                [mysess.run((self.labels_distribution.probs), feed_dict={self.handle: new_handle}) for _ in
                 range(self.num_monte_carlo)])

        return probs

    def train(self, mysession, epoch):


        if tf.gfile.Exists(self.model_dir):
            tf.logging.warning("Warning: deleting old log directory at {}".format(self.model_dir))
            tf.gfile.DeleteRecursively(self.model_dir)

        tf.gfile.MakeDirs(self.model_dir)

        mysession.run(self.init_op)

        my_path = os.path.abspath(os.path.dirname("__file__"))
        save_path = os.path.join(my_path, self.model_dir)

        saver = tf.train.Saver(save_relative_paths=True)
        train_handle = mysession.run(self.training_iterator.string_handle())

        # handle =
        for step in range(epoch):
            _ = mysession.run([self.train_op, self.accuracy_update_op], feed_dict={self.handle: train_handle})
            if step % 100 == 0:
                loss_value, accuracy_value = mysession.run([self.elbo_loss, self.accuracy],
                                                           feed_dict={self.handle: train_handle})
                print("Step: {:>3d} Loss: {:.3f} Training Accuracy: {:.3f}".format(step, loss_value, accuracy_value))

        save_path = os.path.join(save_path, "bnn.epoch.{}.ckpt".format(epoch))

        saver.save(mysession, save_path)



if __name__=="__main__":


    hub_module = "https://tfhub.dev/google/universal-sentence-encoder/2"
    #os.environ["TFHUB_CACHE_DIR"]='/var/folders/8j/wh144j4j5d5c_rq3nbtftnk40000gn/T/tfhub_modules'
    embed=hub.Module(hub_module)

    # M, I, G = LoadPredictor(False)
    # #
    # with tf.Session(graph=G) as sess_1:
    #
    #     M.train(sess_1, 500)


    #
    # Step 2:
    handler_session = tf.Session()
    handler_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #
    M, I, G, S = LoadPredictor(True)
    test_features = USEembeddings("How do I buy ETF?", embed, handler_session)
    #
    #
    probs = M.predict_1(S, test_features)
    #
    print (len(probs[:,0,3]))


    # # Step 3:
    #
    # features = np.load('./datasets/features.npy')
    # sherpa_data_labels = np.load('./datasets/labels.npy')
    #
    # mymodel = Model(features, sherpa_data_labels)
    #
    # mymodel.update_model(500)
    # # with tf.Session() as session:
#     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#     features = session.run(embed(['I will buy.']))
#
#     print(features)