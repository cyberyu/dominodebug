import pandas as pd
import pickle
from bokeh.layouts import layout, row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS, Panel
from bokeh.models.widgets import Div, RangeSlider, Button, DataTable, TableColumn, NumberFormatter, DateFormatter, CheckboxGroup, \
    HTMLTemplateFormatter
from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.client import push_session
from sklearn.svm import SVC
from bokeh.models.widgets import TextInput
from datetime import date
from random import randint
import numpy as np
import os
from os.path import dirname, join
import random
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
import gc
tfd = tfp.distributions
import psutil


global M, S

# visualization of probabilities outcome.   inum is the row id of the results to be shown, and question_text is the original question array
def vis(probs, inum=None, question_text=None):
   class_captions = ['BuyVgMutualFunds', 'HowToExchange', 'SellVgFunds', 'BankAuthenticationTiming', 'AutoTransact',
                     'BuyEtfs', 'RMD', 'QCD', 'ClosedFunds']

   plt.rcParams['figure.figsize'] = [20,10]
   f, ((ax0,ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8,ax9)) = plt.subplots(2,5, sharey=False)

   if_norm_hist = False

   if_kde = False


   if inum is None:
       pos = 0
       dt = 'None'
   else:
       pos = inum
       dt = question_text[inum]

   f.suptitle(dt)

   v = probs[:,pos,0]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax0)
   ax0.set_title("Posterior " + str(class_captions[0]))

   v = probs[:,pos,1]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax1)
   ax1.set_title("Posterior " + str(class_captions[1]))

   v = probs[:,pos,2]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax2)
   ax2.set_title("Posterior " + str(class_captions[2]))

   v = probs[:,pos,3]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax3)
   ax3.set_title("Posterior " + str(class_captions[3]))

   v = probs[:,pos,4]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax4)
   ax4.set_title("Posterior " + str(class_captions[4]))

   v = probs[:,pos,5]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax5)
   ax5.set_title("Posterior " + str(class_captions[5]))

   v = probs[:,pos,6]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax6)
   ax6.set_title("Posterior " + str(class_captions[6]))

   v = probs[:,pos,7]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax7)
   ax7.set_title("Posterior " + str(class_captions[7]))

   v = probs[:,pos,8]
   sns.distplot(v, norm_hist=if_norm_hist, kde=if_kde,  ax=ax8)
   ax8.set_title("Posterior " + str(class_captions[8]))

   # v = probs[:,1,9]
   # sns.distplot(v, norm_hist=True, kde=True,  ax=ax9)
   # ax9.set_title("posterior samples of class " + str(class_captions[9]))
   f.subplots_adjust(hspace=0.5)
   plt.show()



global handler_session


def LoadPredictor(ifrestore=True):



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


def active_learning_tab():
    global M, S

    def USEembeddings(msg, myeb):
        global handler_session

        with tf.Session() as handler_session:
            handler_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            return handler_session.run(myeb([msg]))


    def style(p):
        # Title
        p.title.align = 'center'
        p.title.text_font_size = '10pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '10pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '10pt'
        p.yaxis.axis_label_text_font_style = 'bold'

        # Tick labels
        p.xaxis.major_label_text_font_size = '8pt'
        p.yaxis.major_label_text_font_size = '8pt'

        return p


    def make_plot_data(h):

        my_path = os.path.abspath(os.path.dirname(__file__))

        #print ('h is ' + str(h))
        #print('h type is ' + str(type(h)))

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)  # in byte


        # plot the histogram of y_test_pred_all_denormalized
        arr_hist, edges = np.histogram(h, density=True, bins=20)
        # arr_hist = arr_hist/np.sum(arr_hist)

        #		print ('arr_hist value is ' + str(arr_hist) + ' and sum is '+ str(np.sum(arr_hist)))

        by_carrier = pd.DataFrame(
            columns=['proportion', 'left', 'right', 'f_proportion', 'f_interval', 'name', 'color'])

        arr_df = pd.DataFrame({'proportion': arr_hist, 'left': edges[:-1], 'right': edges[1:]})

        # Format the proportion
        arr_df['f_proportion'] = ['%0.5f' % proportion for proportion in arr_df['proportion']]

        # Format the interval
        arr_df['f_interval'] = ['%d to %d minutes' % (left, right) for left, right in
                                zip(arr_df['left'], arr_df['right'])]

        # Assign the carrier for labels
        arr_df['name'] = 'Try_'
        arr_df['color'] = 'blue'

        by_carrier = by_carrier.append(arr_df)

        print('arr_hist type is ' + str(type(arr_hist)))

        # return ColumnDataSource(by_carrier), arr_df, y_test_pred_mean, y_var_mat_square_mean, arr_hist, edges, pdf_src, pdf_data
        return ColumnDataSource(by_carrier)


    def make_new_plot(h_src, fig_title):

        p = figure(plot_width=300, plot_height=300,
                   title=fig_title,
                   x_axis_label='Probability Score', y_axis_label='Density')

        p.quad(source=h_src, bottom=0, top='proportion', left='left', right='right',
               color='color', fill_alpha=0.7, hover_fill_color='color', hover_fill_alpha=1.0, line_color='black')

        p = style(p)

        return p

    def draw_simulation():
        global handler_session

        global M, S

        inc_msg = div_input.value

        msg_features = USEembeddings(inc_msg, embed)

        probs = M.predict_1(S, msg_features)

        print ('probs is ' + str(probs[:,0,3]))

        hist_src_0 = make_plot_data(probs[:, 0, 0])
        hist_src_1 = make_plot_data(probs[:, 0, 1])
        hist_src_2 = make_plot_data(probs[:, 0, 2])
        hist_src_3 = make_plot_data(probs[:, 0, 3])
        hist_src_4 = make_plot_data(probs[:, 0, 4])
        hist_src_5 = make_plot_data(probs[:, 0, 5])
        hist_src_6 = make_plot_data(probs[:, 0, 6])
        hist_src_7 = make_plot_data(probs[:, 0, 7])
        hist_src_8 = make_plot_data(probs[:, 0, 8])

        h_src_0.data.update(hist_src_0.data)
        h_src_1.data.update(hist_src_1.data)
        h_src_2.data.update(hist_src_2.data)
        h_src_3.data.update(hist_src_3.data)
        h_src_4.data.update(hist_src_4.data)
        h_src_5.data.update(hist_src_5.data)
        h_src_6.data.update(hist_src_6.data)
        h_src_7.data.update(hist_src_7.data)
        h_src_8.data.update(hist_src_8.data)

        source.data = {
            'p0' :          [str("{:.2f}".format(np.mean(probs[:, 0, 0])))+'('+"{:.2f}".format(np.std(probs[:, 0, 0]))+')'],
            'p1'         :  [str("{:.2f}".format(np.mean(probs[:, 0, 1])))+'('+"{:.2f}".format(np.std(probs[:, 0, 1]))+')'],
            'p2'         :  [str("{:.2f}".format(np.mean(probs[:, 0, 2])))+'('+"{:.2f}".format(np.std(probs[:, 0, 2]))+')'],
            'p3'  :         [str("{:.2f}".format(np.mean(probs[:, 0, 3])))+'('+"{:.2f}".format(np.std(probs[:, 0, 3]))+')'],
            'p4'  :         [str("{:.2f}".format(np.mean(probs[:, 0, 4])))+'('+"{:.2f}".format(np.std(probs[:, 0, 4]))+')'],
            'p5'  :         [str("{:.2f}".format(np.mean(probs[:, 0, 5])))+'('+"{:.2f}".format(np.std(probs[:, 0, 5]))+')'],
            'p6'  :         [str("{:.2f}".format(np.mean(probs[:, 0, 6])))+'('+"{:.2f}".format(np.std(probs[:, 0, 6]))+')'],
            'p7':           [str("{:.2f}".format(np.mean(probs[:, 0, 7])))+'('+"{:.2f}".format(np.std(probs[:, 0, 7]))+')'],
            'p8':           [str("{:.2f}".format(np.mean(probs[:, 0, 8])))+'('+"{:.2f}".format(np.std(probs[:, 0, 8]))+')']
        }

        del msg_features, probs, M, I, G, S
        handler_session.close()
        collected = gc.collect()
        del collected


    hub_module = "https://tfhub.dev/google/universal-sentence-encoder/2"
    os.environ["TFHUB_CACHE_DIR"]='./tfhub_modules'
    embed=hub.Module(hub_module)

    M, I, G, S = LoadPredictor(True)


    data = dict(
        p1=[0],
        p2=[0],
        p3=[0],
        p4=[0],
        p5=[0],
        p6=[0],
        p7=[0],
        p8=[0],
        p9=[0]

    )
    source = ColumnDataSource(data)

    columns = [
        TableColumn(field="p0", title="BuyMF"),
        TableColumn(field="p1", title="H2Ex"),
        TableColumn(field="p2", title="SellMF"),
        TableColumn(field="p3", title="Bank"),
        TableColumn(field="p4", title="AutoTrans"),
        TableColumn(field="p5", title="BuyETF"),
        TableColumn(field="p6", title="RMD"),
        TableColumn(field="p7", title="QCD"),
        TableColumn(field="p8", title="ClosedFd")

    ]

    data_table = DataTable(source=source, columns=columns, width=1000, height=100)


    # df_question = pickle.load(open("df_question.pkl", "rb"))
    #
    # df_question["p0"] = ""
    # df_question["p1"] = ""
    # df_question["p2"] = ""
    # df_question["p3"] = ""
    # df_question["p4"] = ""
    # df_question["p5"] = ""
    # df_question["p6"] = ""
    #
    # # find the indicies of -1
    #
    # # i_test = df_question.index[df_question['Labels'] == -1].tolist()
    # # i_train = df_question.index[df_question['Labels'] != -1].tolist()
    #
    # #current_train = df_question.iloc[i_train]
    # current_test = df_question.iloc[0]
    #
    #
    # columns_test = [
    #     TableColumn(field="p0", title="p0", formatter=NumberFormatter(format="0.00"), width=50),
    #     TableColumn(field="p1", title="p1", formatter=NumberFormatter(format="0.00"), width=50),
    #     TableColumn(field="p2", title="p2", formatter=NumberFormatter(format="0.00"), width=50),
    #     TableColumn(field="p3", title="p3", formatter=NumberFormatter(format="0.00"), width=50),
    #     TableColumn(field="p4", title="p4", formatter=NumberFormatter(format="0.00"), width=50),
    #     TableColumn(field="p5", title="p5", formatter=NumberFormatter(format="0.00"), width=50),
    #     TableColumn(field="p6", title="p6", formatter=NumberFormatter(format="0.00"), width=50)
    # ]

    button = Button(label="Predict", button_type="success", width=100)
    button.on_click(draw_simulation)

    # source_test = ColumnDataSource(data=dict())
    #
    # data_table_test = DataTable(source=source_test, columns=columns_test, editable=False, height=400, width=400,
    #                             fit_columns=False)
    #

    # def update(v0, v1, v2, v3, v4, v5, v6, v7, v8):
    #     # current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
    #     # current = df
    #
    #     source_test.data = {
    #         'p0': v0,
    #         'p1': v1,
    #         'p2': v2,
    #         'p3': v3,
    #         'p4': v4,
    #         'p5': v5,
    #         'p6': v6,
    #         'p7': v7,
    #         'p8': v8
    #     }


    div_input = TextInput(value="", title="Enter your question now: ", width=1000)
    #show(widgetbox(div_input))

    # div_train = Div(text="""Training Data""", width=100, height=10)
    # div_test = Div(text="""Test Data""", width=100, height=10)
    #
    # doc_layout = layout([[column(row(column(widgetbox(div_train), widgetbox(button)), widgetbox(data_table_train)),
    #                              row(widgetbox(div_test), widgetbox(data_table_test)))]], sizing_mode='scale_width')
    h_src_0 = make_plot_data(2)
    h_src_1 = make_plot_data(2)
    h_src_2 = make_plot_data(2)
    h_src_3 = make_plot_data(2)
    h_src_4 = make_plot_data(2)
    h_src_5 = make_plot_data(2)
    h_src_6 = make_plot_data(2)
    h_src_7 = make_plot_data(2)
    h_src_8 = make_plot_data(2)


    p0 = make_new_plot(h_src_0, "BuyVgMutualFunds")
    p0 = style(p0)

    p1 = make_new_plot(h_src_1, "HowToExchange")
    p1 = style(p1)

    p2 = make_new_plot(h_src_2, "SellVgFunds")
    p2 = style(p2)

    p3 = make_new_plot(h_src_3, "BankAuthenticationTiming")
    p3 = style(p3)

    p4 = make_new_plot(h_src_4, "AutoTransact")
    p4 = style(p4)

    p5 = make_new_plot(h_src_5, "BuyEtfs")
    p5 = style(p5)

    p6 = make_new_plot(h_src_6, "RMD")
    p6 = style(p6)

    p7 = make_new_plot(h_src_7, "QCD")
    p7 = style(p7)

    p8 = make_new_plot(h_src_8, "ClosedFunds")
    p8 = style(p8)

    #row(column([[p1, p2, p3, p4, p5], [p6, p7, p8, p9]]



    doc_layout = layout(row(column(widgetbox(div_input), widgetbox(button))),row(widgetbox(data_table)),row([p0,p1,p2,p3,p4]), row([p5,p6,p7,p8]), sizing_mode='scale_width')

    # doc_layout = layout([
    #     [div_input],
    #     [button],
    #     p2,
    # ], sizing_mode='stretch_both')

    tab = Panel(child=doc_layout, title='PredictLabel')

    #update("1","2","3","4","5","6","7","8","9")

    return tab
