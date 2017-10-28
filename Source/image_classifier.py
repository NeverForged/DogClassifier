import numpy as np
import tensorflow as tf
import pickle as pickle
# import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


class ImageClassifier(BaseEstimator, ClassifierMixin):
    """Image Classifier.
    Uses a fairly standard Convoluted Neural Netwrok architecture to
    classify some number of objects.

    Parameters
    ----------
    picsize:  Size of one side of square image(s) to be classified in
              pixels
    classes:  List of class names for your classes
    convolution_size: (= 5 by default) The size of the convolutions, all
              convolutions are convolution_size by convolution_size.
    epochs:   (= 100 by default) How many times it runs thru the entire dataset
    out_channels: (= 24 by default) The number of Output channels in the
              first convolution.
    out_channels_2: (= 48 by default) The number of Output channels in the
              second convolution.
    init:     (= 'he' by default) This is the initializer, either 'he' for He
              initialization (sqrt(2/N)) or 'x' for Xavier (sqrt(1/N)).  Can
              also be a number, such as '2.0' for He or '1.0' for Xavier
    hidden_units: (= 512 by default) Number of hidden features between the
              convolutions and the output
    regularization_strength: (= 1.0 by default) The factor used in
              regularization step (see tf.nn.l2_loss)
    batch_size: (= 100 by default) Number of images used in each training step.
    learning_rate (= 0.001 by default) The starting learning rate of the Atom
              Optimizer, see tf.train.AdamOptimizer() for details.
    pool_size: (= 2 by default) The size of the pools.
    verbose: (= False by default) Set to true to get an update on percentage
             done and training Accuracy.
    loss_threshold: (= 0.001 by default) An Early Stop parameter, determines
              the minimum difference between loss reads before the epochs kick
              out early.
    grid_search: (=False by default) set True to close the session as soon as
              a score is generated.

    Attributes
    ----------
    W1:        Values of convolution 1.  Initialized as a truncated-normal
               w/stddev=0.1 (tensorflow object).  Can be set at initialization.
    b1:        Bias terms for convolution 1 (tensorflow object)  Can be set at
               initialization
    W1:        Values of convolution 2.  Initialized as a truncated-normal
               w/stddev=0.1 (tensorflow object).  Can be set at initialization.
    b1:        Bias terms for convolution 2 (tensorflow object)  Can be set at
               initialization
    Wf:        Values of Fully Connected network layer.  Initialized as a
               truncated-normal w/stddev=0.1 (tensorflow object).  Can be set at
               initialization.
    bf:        Bias terms for  Fully Connected network layer (tensorflow object)
               Can be set at initialization
    Wf2:       Values of output layer.  Initialized as a truncated-normal
               w/stddev=0.1 (tensorflow object).  Can be set at initialization.
    bf2:       Bias terms for output layer (tensorflow object)  Can be set at
               initialization

    Notes
    -----

    """

    def __init__(self, picsize=32, classes=[0, 1], convolution_size=5,
                 training_epochs=int(100), out_channels=24, out_channels_2=48,
                 init='he', grid_search=False, hidden_units=512,
                 regularization_strength=1.0, batch_size=99,
                 learning_rate=0.001, pool_size=2, verbose=False, W1=None,
                 b1=None, W2=None, b2=None, Wf=None, bf=None, Wf2=None,
                 bf2=None, loss_threshold=0.001, load_file=None):
        '''
        Initializer.
        '''
        try:
            d = pickle.load(open(load_file, "rb"))
            self.picsize = d['picsize']
            self.classes = d['classes']
            self.convolution_size = int(d['convolution_size'])
            self.training_epochs = int(d['training_epochs'])
            self.out_channels = int(d['out_channels'])
            self.out_channels_2 = int(d['out_channels_2'])
            self.hidden_units = int(d['hidden_units'])
            self.regularization_strength = float(d['regularization_strength'])
            self.batch_size = d['batch_size']
            self.learning_rate = float(d['learning_rate'])
            self.pool_size = int(d['pool_size'])
            self.verbose = d['verbose']
            self.grid_search = d['grid_search']
            self.W1_best = d['W1_best']
            self.b1_best = d['b1_best']
            self.W2_best = d['W2_best']
            self.b2_best = d['b2_best']
            self.Wf_best = d['Wf_best']
            self.Wf2_best = d['Wf2_best']
            self.bf_best = d['bf_best']
            self.bf2_best = d['bf2_best']
            self.init_factor = d['init_factor']
            self.loss_threshold = float(d['loss_threshold'])
            self.train_accuracies = d['train_accuracies']
            self.val_accuracies = d['val_accuracies']
            self.loss_function = d['loss_function']
        except:
            self.picsize = int(picsize)
            self.classes = classes
            self.convolution_size = int(convolution_size)
            self.training_epochs = int(training_epochs)
            self.out_channels = int(out_channels)
            self.out_channels_2 = int(out_channels_2)
            self.hidden_units = int(hidden_units)
            self.regularization_strength = float(regularization_strength)
            self.batch_size = batch_size
            self.learning_rate = float(learning_rate)
            self.pool_size = int(pool_size)
            self.verbose = verbose
            self.grid_search = grid_search
            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
            self.Wf = Wf
            self.Wf2 = Wf2
            self.bf = bf
            self.bf2 = bf2
            if init == 'he':
                self.init_factor = 2.0
            elif init == 'x':
                self.init_factor = 1.0
            else:
                try:
                    self.init_factor = float(init)
                except:
                    self.init_factor = 2.0
            self.loss_threshold = float(loss_threshold)

    def MakeCNN(self, fitting_this=False):
        '''
        Make the actual CNN...
        '''
        self.initializer = tf.contrib.layers.variance_scaling_initializer(
                            factor=self.init_factor)
        # ---------- Convolutional layer 1 ----------
        # third number = channels, so 3
        try:
            self.W1 = tf.get_variable('W1', init=tf.constant(self.W1))
        except:
            self.W1 = tf.Variable(self.initializer([self.convolution_size,
                                                    self.convolution_size, 3,
                                                    self.out_channels]),
                                  name='W1')
        try:
            self.b1 = tf.get_variable('b1', init=tf.constant(self.b1))
        except:
            self.b1 = tf.Variable(self.initializer([self.out_channels]),
                                  name='b1')

        # Tensors representing our input dataset and our input labels
        self.x = tf.placeholder(tf.float32, shape=[None,
                                                   self.picsize*self.picsize*3],
                                name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, len(self.classes)],
                                name='y')

        # When passing through the network, we want to represent the images
        # as a 4-tensor: (n_images, image_height, image_width, n_channels)
        self.x_image = tf.reshape(self.x, [-1, self.picsize, self.picsize, 3],
                                  name='ximage')

        # now the convolution, turns W1/x_m
        '''Given an input tensor of shape
        [batch, in_height, in_width, in_channels]
        and a filter / kernel tensor of shape
        [filter_height, filter_width, in_channels, out_channels]'''
        self.conv1 = tf.nn.conv2d(self.x_image, self.W1,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME', name='conv1')
        # so [?, out_height, out_width, 75]
        self.relu1 = tf.nn.relu(self.conv1 + self.b1)
        # shape [?, picsize, picsize, out_channels]

        # ----------  Max Pool 'Layer' 1  ----------
        # ksize is the size of the windows to overlay.
        # strides controls how to slide the windows around the image.
        self.layer_1_out = tf.nn.max_pool(self.relu1,
                                          ksize=[1, self.pool_size,
                                                 self.pool_size, 1],
                                          strides=[1, self.pool_size,
                                                  self.pool_size, 1],
                                          padding='SAME')
        # size = [?, picsize/2, picsize/2, out_channles]

        # ---------- Convolutional layer  2 ----------
        try:
            self.W2 = tf.get_variable('W2', init=tf.constant(self.W2))
        except:
            self.W2 = tf.Variable(self.initializer([self.convolution_size,
                                                    self.convolution_size,
                                                    self.out_channels,
                                                    self.out_channels_2]),
                                                    name='w2')
        try:
            self.b2 = tf.get_variable('b2', init=tf.constant(self.b2))
        except:
            self.b2 = tf.Variable(self.initializer([self.out_channels_2]),
                                  name='b2')

        # now the convolution... on layer_1_out
        self.conv2 = tf.nn.conv2d(self.layer_1_out, self.W2,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME', name='conv2')
        self.relu2 = tf.nn.relu(self.conv2 + self.b2)
        # size of both above: (?, picsize/2, picsize/2, out_channels_2)

        # ---------- Max Pool 'Layer' 2 ----------
        self.layer_2_out = tf.nn.max_pool(self.relu2,
                                          ksize=[1, self.pool_size,
                                                 self.pool_size, 1],
                                          strides=[1, self.pool_size,
                                                   self.pool_size, 1],
                                          padding='SAME')
        # size of both above: (?, picsize/4, picsize/4, out_channels_2)

        # ---------- Fully Connected layer ----------
        try:
            self.Wf = tf.get_variable('Wf', init=tf.constant(self.Wf))
        except:
            self.Wf = tf.Variable(self.initializer([int(self.picsize**2 *
                                                   1/(self.pool_size**4) *
                                                   self.out_channels_2),
                                                   self.hidden_units]),
                                                   name = 'Wf')

        try:
            self.bf = tf.get_variable('bf', init=tf.constant(self.bf))
        except:
            self.bf = tf.Variable(self.initializer([self.hidden_units]),
                                  name = 'bf')

        # Flatten the output of the second layer.  This allows us to do
        # a simple matrix multiplication with the weight matrix for the
        # fully connected layer.
        self.layer_2_out_flat = tf.reshape(
                                    self.layer_2_out, [-1,
                                                       int(self.picsize**2 *
                                                       1/(self.pool_size**4) *
                                                       self.out_channels_2)])

        self.fully_connected_1_out = tf.nn.relu(tf.matmul(self.layer_2_out_flat,
                                                          self.Wf) + self.bf)
        # size: ?, hidden_units

        # ---------- The Output Layer ----------
        try:
            self.Wf2 = tf.get_variable('Wf2', init=tf.constant(self.Wf2))
        except:
            self.Wf2 = tf.Variable(self.initializer([self.hidden_units,
                                                     len(self.classes)]),
                                                    name = 'Wf2')
        try:
            self.bf2 = tf.get_variable('bf2', init=tf.constant(self.bf2))
        except:
            self.bf2 = tf.Variable(self.initializer([len(self.classes)]),
                                   name = 'bf2')

        # Predictions, but on a log-odds scale.
        self.fully_connected_2_out = tf.matmul(self.fully_connected_1_out,
                                               self.Wf2) + self.bf2
        # shape: inputs, # classes

        # ---------- Training & Loss Function ----------
        # The basic loss function, cross entropy.
        self.cross_entropy = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(
                                    labels=self.y,
                                    logits=self.fully_connected_2_out))

        # Use L2 regularization on all the weights in the network.

        self.regularization_term = self.regularization_strength * (
                                    tf.nn.l2_loss(self.W1) +
                                    tf.nn.l2_loss(self.W2) +
                                    tf.nn.l2_loss(self.Wf) +
                                    tf.nn.l2_loss(self.Wf2))

        # The total loss function we will minimize is cross entropy
        # plus regularization.
        self.total_loss = self.cross_entropy + self.regularization_term

        # Create a tensor to track the accuracy during training.
        self.correct_prediction = tf.equal(tf.argmax(self.fully_connected_2_out,
                                                     1),
                                           tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                               tf.float32))


        # Training...
        # ADAM is a sophisticated version of gradient descent that adapts the
        # learning rate over time.
        if fitting_this:
            self.train_step = (tf.train.AdamOptimizer(self.learning_rate)
                                        .minimize(self.total_loss))


        # Set up the session...
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, y):
        '''
        Runs the CNN for X according to y.

        Parameters
        ----------
        X:   Input data, as a flattened array of pictures (N, picsize*picsize*3)
        y:   Input classes (N, # of items in classes)

        Attributes
        ----------
        train_accuracies: A list of train accuracies at each 'epoch'.
        '''
        self.MakeCNN(True)
        self.train_accuracies = []
        self.loss_function = []
        self.val_accuracies = []

        # validation set is roughly 20% of data, plus a little to avoid
        # sparse batches
        batch_steps = int((X.shape[0]*0.8)/self.batch_size)
        extra = X.shape[0] - self.batch_size*batch_steps
        Xval = X[:extra]
        X = X[extra:]
        yval = y[:extra]
        y = y[extra:]
        if self.verbose==True:
            print('\rPercent Complete: --.----% - ' +
                  'Train Accuracy: --.--- - Validation Accuracy: ' +
                  '--.--- - Loss Function: ----.----', end='')

        j = 0
        old = -9000
        dif = 90001 # it's over 9000!
        self.best_val_acc_ = 0.

        while (int(j) < int(self.training_epochs) and
               float(abs(dif)) > float(self.loss_threshold)):
            lst_acc = []
            # shuffle data to build self.batch_size...
            Xhold = X.copy()
            Yhold = y.copy()
            new = np.array([i for i in range(Xhold.shape[0])])
            np.random.shuffle(new)
            for i, n in enumerate(new):
                X[i, :] = Xhold[n, :]
                y[i, :] = Yhold[n, :]
            # run thru the entire training set...
            loss_temp = []
            for i in range(batch_steps):
                # make sure we have enough self.batch_size for a full batch
                xbatch = X[i*self.batch_size:
                           i*self.batch_size + self.batch_size - 1, :]
                ybatch = y[i*self.batch_size:
                           i*self.batch_size + self.batch_size - 1, :]
                self.train_step.run(feed_dict={self.x: xbatch,
                                               self.y: ybatch})
                train_accuracy = self.accuracy.eval(feed_dict=
                                                        {self.x:xbatch,
                                                         self.y: ybatch})
                lst_acc.append(train_accuracy)
                loss = self.total_loss.eval(feed_dict = {self.x:xbatch,
                                                         self.y: ybatch})
                loss_temp.append(loss)
                # make sure it's working...
                if self.verbose == True:
                    calc = (100.0*((j*batch_steps + i)/
                            (self.training_epochs*batch_steps)))
                    if len(self.loss_function) >= 1:
                        print('\rPercent Complete: {:.4f}% - '.format(calc) +
                              'Train Accuracy: {:.3f}% '
                              .format(100*self.train_accuracies[-1]) +
                              '- Validation Accuracy: {:.3f}% - '
                              .format(self.val_accuracies[-1] * 100) +
                              'Loss Function: {:.4f}'
                              .format(self.loss_function[-1]), end='')
                    else:
                        print('\rPercent Complete: {:.4f}% - '.format(calc) +
                              'Train Accuracy: --.--- - Validation Accuracy: ' +
                              '--.--- - Loss Function: ----.----', end='')
            # update
            j += 1
            # Print out diagnostics
            self.train_accuracies.append(np.mean(train_accuracy))
            self.loss_function.append(np.mean(loss_temp))
            loss_temp = []
            acc = self.accuracy.eval(feed_dict= {self.x:Xval, self.y:yval})
            self.val_accuracies.append(acc)
            if acc >= self.best_val_acc_:
                self.best_val_acc_
                self.W1_best = self.W1.eval()
                self.b1_best = self.b1.eval()
                self.W2_best = self.W2.eval()
                self.b2_best = self.b2.eval()
                self.Wf_best = self.Wf.eval()
                self.bf_best = self.bf.eval()
                self.Wf2_best = self.Wf2.eval()
                self.bf2_best = self.bf2.eval()
            dif = self.loss_function[-1] - old
            old = self.loss_function[-1]
            if self.verbose == True:
                print('\rPercent Complete: {:.4f}% - Train Accuracy: {:.3f}% '
                      .format(calc, 100*self.train_accuracies[-1]) +
                      '- Validation Accuracy: {:.3f}% - Loss Function: {:.4f}'
                      .format(self.val_accuracies[-1] * 100,
                              self.loss_function[-1]),
                       end='')
        # fix loss function...
        self.loss_function = self.loss_function/np.max(self.loss_function[1:])
        self.sess.run(tf.assign(self.W1, self.W1_best))
        self.sess.run(tf.assign(self.W2, self.W2_best))
        self.sess.run(tf.assign(self.Wf, self.Wf_best))
        self.sess.run(tf.assign(self.Wf2, self.Wf2_best))
        self.sess.run(tf.assign(self.b1, self.b1_best))
        self.sess.run(tf.assign(self.b2, self.b2_best))
        self.sess.run(tf.assign(self.bf, self.bf_best))
        self.sess.run(tf.assign(self.bf2, self.bf2_best))
        return self

    def reset_(self):
        self.MakeCNN()
        self.sess.run(tf.assign(self.W1, self.W1_best))
        self.sess.run(tf.assign(self.W2, self.W2_best))
        self.sess.run(tf.assign(self.Wf, self.Wf_best))
        self.sess.run(tf.assign(self.Wf2, self.Wf2_best))
        self.sess.run(tf.assign(self.b1, self.b1_best))
        self.sess.run(tf.assign(self.b2, self.b2_best))
        self.sess.run(tf.assign(self.bf, self.bf_best))
        self.sess.run(tf.assign(self.bf2, self.bf2_best))

    def score(self, X, y):
        '''
        Returns mean accuracy of predicting x given y.

        Parameters
        X:   Input data, as a flattened array of pictures (N, picsize*picsize*3)
        y:   Input classes (N, # of items in classes)
        -----------

        '''
        self.reset_()
        score = self.accuracy.eval(feed_dict={self.x:X, self.y:y})
        if self.grid_search:
            self.sess.close()
        return score


    def predict(self, X, y=None):
        '''
        Returns a prediction based on X
        '''
        try:
            # self.reset_()
            return tf.argmax(self.fully_connected_2_out,1).eval(feed_dict = {self.x:X})

        except:
            self.reset_()
            return tf.argmax(self.fully_connected_2_out,1).eval(feed_dict = {self.x:X})

    def predict_proba(self, X, y=None):
        '''
        Returns a probability prediction based on X (log-odds)
        '''
        try:
            # self.reset_()
            return self.fully_connected_2_out.eval(feed_dict = {self.x:X})
        except:
            self.reset_()
            return self.fully_connected_2_out.eval(feed_dict = {self.x:X})

    def save_(self, save_file='models/model.pickle'):
        '''
        Saves all the relevant things in a pickle file for later retrieval.

        tf.train.Saver() failed for some reason, and I have non-tf things to
        save anyhow.
        '''
        self.save_file = save_file
        d = {}
        d['picsize'] = self.picsize
        d['classes'] = self.classes
        d['convolution_size'] = self.convolution_size
        d['training_epochs'] = self.training_epochs
        d['out_channels'] = self.out_channels
        d['out_channels_2'] = self.out_channels_2
        d['hidden_units']  = self.hidden_units
        d['regularization_strength'] = self.regularization_strength
        d['batch_size'] = self.batch_size
        d['learning_rate']  = self.learning_rate
        d['pool_size'] = self.pool_size
        d['verbose'] = self.verbose
        d['grid_search'] = False
        d['W1_best'] = self.W1_best
        d['b1_best'] = self.b1_best
        d['W2_best'] = self.W2_best
        d['b2_best'] = self.b2_best
        d['Wf_best'] = self.Wf_best
        d['Wf2_best'] = self.Wf2_best
        d['bf_best'] = self.bf_best
        d['bf2_best'] = self.bf2_best
        d['init_factor'] = self.init_factor
        d['loss_threshold'] = self.loss_threshold
        d['train_accuracies'] = self.train_accuracies
        d['val_accuracies'] = self.val_accuracies
        d['loss_function'] = self.loss_function

        with open(self.save_file, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    picsize = 100
    lst_dogs=['a', 'b']
    model = ImageClassifier(picsize, lst_dogs,
                             out_channels = 24,
                             out_channels_2 = 48,
                             hidden_units = 100,
                             regularization_strength = 0.01,
                             batch_size = 64,
                             learning_rate = 0.1,
                             convolution_size = 5,
                             pool_size = 2,
                             training_epochs = 5,
                             verbose=True)
    print('okay')
