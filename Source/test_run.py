import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from dog_images import DogImages
from image_classifier import ImageClassifier

def main():
    picsize = 128
    # Step 1: Get List of Dogs
    lst = [x[0] for x in os.walk('../Images')]
    lst_dogs = [a.replace('../Images/', '') for a in lst[1:]]

    dog_images = DogImages(lst_dogs, picsize)
    # dog_images.generate_img_files()
    train_imgs = dog_images.load_images('train')
    test_imgs = dog_images.load_images('test')
    Xtest = test_imgs[0]
    Ytest = test_imgs[1]
    Xtrain = train_imgs[0]
    Ytrain = train_imgs[1]

    for N, dogs in enumerate(lst_dogs):
        best_score = 0
        dog = dogs[dogs.index('-')+1:]
        classes = [dog, 'Not-'+dog]
        for i in range(5):
            # free up some memory...
            tf.reset_default_graph()
            Xtrain, Ytrain = shuffle_function(Xtest, Ytest)
            Xtest, Ytest = shuffle_function(Xtest, Ytest)

            Ytrain_1 = np.zeros((Ytrain.shape[0],2))
            Ytest_1 = np.zeros((Ytest.shape[0],2))

            Ytrain_1[Ytrain[:,N]==1] = [0, 1]
            Ytrain_1[Ytrain[:,N]==0] = [1, 0]

            Ytest_1[Ytest[:,N]==1] = [0, 1]
            Ytest_1[Ytest[:,N]==0] = [1, 0]

            Ytrain_a = Ytrain_1[Ytrain_1[:,0] == 1]
            Xtrain_a = Xtrain[Ytrain_1[:,0] == 1]
            Ytrain_b = Ytrain_1[Ytrain_1[:,0] == 0][:int(1.5*Ytrain_a.shape[0])]
            Xtrain_b = Xtrain[Ytrain_1[:,0] == 0][:int(1.5*Ytrain_a.shape[0])]

            Ytrain_run = np.concatenate((Ytrain_a, Ytrain_b))
            Xtrain_run = np.concatenate((Xtrain_a, Xtrain_b))

            model = ImageClassifier(picsize, classes,
                                out_channels = 12,
                                 out_channels_2 = 24,
                                 hidden_units = 100,
                                 regularization_strength = 1.0,
                                 batch_size = 100,
                                 learning_rate = 0.001,
                                 convolution_size = 5,
                                 pool_size = 2,
                                 training_epochs = 100,
                                 loss_threshold = 0.01,
                                 verbose=True)
            model.fit(Xtrain_run, Ytrain_run)
            score = model.score(Xtest, Ytest_1)

            if score > best_score and score < 1.0:
                model.save_('models/' + dog + '.pickle')
            model.sess.close()
            tf.reset_default_graph()
            del model
            print()
            print('\rOverall Percent Complete: {:.4f}%'
                    .format((i + 5*N)/(120.0*5)))

def shuffle_function(X, y):
    Xhold = X.copy()
    Yhold = y.copy()
    new = np.array([i for i in range(Xhold.shape[0])])
    np.random.shuffle(new)
    for i, n in enumerate(new):
        X[i, :] = Xhold[n, :]
        y[i, :] = Yhold[n, :]
    return X, y

if __name__ == '__main__':
    main()
