import tensorflow as tf

class ImageClassifier(object):
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
    runs:     (= 100 by default) How many times it runs before giving up,
              higher is better.
    out_channels: (= 24 by default) The number of Output channels in the
              first convolution.
    out_channels_2: (= 48 by default) The number of Output channels in the
              second convolution.
    accuracy_thresh: (= 0.5 by default) the difference in threshold required
              before it stops running that batch.
    hidden_units: (= 512 by default) Number of hidden features between the
              convolutions and the output
    regularization_strength: (= 1.0 by default) The factor used in
              regularization step (see tf.nn.l2_loss)
    slides: (= 50 by default) Number of pictures used in each training step.
    learning_rate (= 0.001 by default) The starting learning rate of the Atom
              Optimizer, see tf.train.AdamOptimizer() for details.
    pool_size: (= 2 by default) The size of the pools.
    verbose: (= False by default) Set to true to get an update on percentage
             done and training Accuracy.

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

    def __init__(self, picsize, classes, convolution_size=5, runs=100,
                 out_channels=24, out_channels_2=48, accuracy_thresh=0.5,
                 hidden_units=512, regularization_strength=1.0, slides=50,
                 learning_rate=0.001, pool_size=2, accuracy_counter=5,
                 verbose=False, W1=None, b1=None, W2=None, b2=None, Wf=None,
                 bf=None, Wf2=None, bf2=None):
        '''
        Initializer.
        '''
        self.picsize = picsize
        self.classes = classes
        self.convolution_size = convolution_size
        self.runs = runs
        self.out_channels = out_channels
        self.out_channels_2 = out_channels_2
        self.accuracy_thresh=accuracy_thresh
        self.hidden_units = hidden_units
        self.regularization_strength = regularization_strength
        self.slides = slides
        self.learning_rate = learning_rate
        self.pool_size = pool_size
        self.accuracy_counter=5
        self.verbose = verbose

        # ---------- Convolutional layer 1 ----------
        # third number = channels, so 3
        try:
            self.W1 = tf.constant(W1)
        except:
            self.W1 = tf.Variable(tf.truncated_normal(shape=[self.convolution_size,
                                                             self.convolution_size,
                                                             3, out_channels],
                                                      stddev=0.1), name='W1')
        try:
            self.b1 = tf.constant(b1)
        except:
            self.b1 = tf.Variable(tf.constant(0.1, shape=[out_channels]),
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
            self.W2 = tf.constant(W2)
        except:
            self.W2 = tf.Variable(tf.truncated_normal(shape=[
                                                         self.convolution_size,
                                                         self.convolution_size,
                                                         self.out_channels,
                                                         self.out_channels_2],
                                                  stddev=0.1), name='w2')
        try:
            self.b2 = tf.constant(b2)
        except:
            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.out_channels_2]))

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
            self.Wf = tf.constant(W2)
        except:
            self.Wf = (tf.Variable(
                       tf.truncated_normal(shape=[int(self.picsize**2 *
                                                   1/16 * self.out_channels_2),
                                                   self.hidden_units],
                                                   stddev=0.01)))
        try:
            self.bf = tf.constant(bf)
        except:
            self.bf = tf.Variable(tf.constant(0.1, shape=[self.hidden_units]))

        # Flatten the output of the second layer.  This allows us to do
        # a simple matrix multiplication with the weight matrix for the
        # fully connected layer.
        self.layer_2_out_flat = tf.reshape(
                                    self.layer_2_out, [-1,
                                                       int(self.picsize**2 *
                                                       1/16 *
                                                       self.out_channels_2)])

        self.fully_connected_1_out = tf.nn.relu(tf.matmul(self.layer_2_out_flat,
                                                          self.Wf) + self.bf)
        # size: ?, hidden_units

        # ---------- The Output Layer ----------
        try:
            self.Wf2 = tf.constant(Wf2)
        except:
            self.Wf2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_units,
                                                          len(self.classes)],
                                                          stddev=0.01))
        try:
            self.bf2 = tf.constant(bf2)
        except:
            self.bf2 = tf.Variable(tf.constant(0.1, shape=[len(self.classes)]))

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
        self.train_accuracies = []
        training_epochs = int((X.shape[0])/self.slides)
        if self.verbose==True:
            print('{} Slides per epoch for {} training epochs'
                   .format(self.slides, training_epochs))
            print('\rPercent Complete: {:.2f}% - Accuracy: {:.2f}%'
                  .format(0, 0), end='')
        slides = self.slides
        for i in range(training_epochs):
            # while i <= training_epochs and not math.isnan(last_accuracy):
            xbatch = X[i*slides:i*slides + slides - 1]
            ybatch = y[i*slides:i*slides + slides - 1]
            steps = 0
            train_accuracy = 0.0
            dif = 10*self.accuracy_thresh
            while dif > self.accuracy_thresh or steps <= self.runs:
                old = train_accuracy
                self.train_step.run(feed_dict={self.x: xbatch, self.y: ybatch})
                train_accuracy = self.accuracy.eval(feed_dict={self.x:xbatch,
                                                               self.y: ybatch})
                dif = train_accuracy - old
                steps += 1

            # Print out diagnostics
            self.train_accuracies.append(train_accuracy)
            if self.verbose == True:
                print('\rPercent Complete: {:.1f}% - Train Accuracy: {:.1f}%'
                      .format(100.0*float((i+1)/training_epochs),
                              100*self.train_accuracies[-1]), end='')

    def score(self, X, y):
        '''
        Returns mean accuracy of predicting x given y.

        Parameters
        X:   Input data, as a flattened array of pictures (N, picsize*picsize*3)
        y:   Input classes (N, # of items in classes)
        -----------

        '''
        return (self.accuracy
                .eval(feed_dict={self.x:X, self.y:y}))

    def predict(self, X):
        '''
        Returns a prediction based on X
        '''
        return (tf.argmax(self.fully_connected_2_out, 1)
                .eval(feed_dict = {self.x:X}))

    def predict_proba(self, X):
        '''
        Returns a probability prediction based on X (log-odds)
        '''
        return self.fully_connected_2_out.eval(feed_dict = {self.x:X})
