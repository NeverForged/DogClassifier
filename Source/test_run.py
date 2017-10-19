import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from dog_images import DogImages
from image_classifier import ImageClassifier
from sklearn.model_selection import GridSearchCV


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

    Xtrain, Ytrain = shuffle_function(Xtest, Ytest)
    Xtest, Ytest = shuffle_function(Xtest, Ytest)

    N = 16
    # need 0:1 to get 0... it's a weird numpy thing
    # so for lst_dogs[N], go Yt...[:, N:N + 1]
    Ytrain_1 = np.zeros((Ytrain.shape[0],2))
    Ytest_1 = np.zeros((Ytest.shape[0],2))

    Ytrain_1[Ytrain[:,N]==1] = [1, 0]
    Ytrain_1[Ytrain[:,N]==0] = [0, 1]

    Ytest_1[Ytest[:,N]==1] = [1, 0]
    Ytest_1[Ytest[:,N]==0] = [0, 1]

    Ytrain_a = Ytrain_1[Ytrain_1[:,0] == 1]
    Xtrain_a = Xtrain[Ytrain_1[:,0] == 1]
    Ytrain_b = Ytrain_1[Ytrain_1[:,0] == 0][:3*Ytrain_a.shape[0]]
    Xtrain_b = Xtrain[Ytrain_1[:,0] == 0][:3*Ytrain_a.shape[0]]

    Ytrain_run = np.concatenate((Ytrain_a, Ytrain_b))
    Xtrain_run = np.concatenate((Xtrain_a, Xtrain_b))

    lst_dogs_1 = [lst_dogs[N], 'None']

    model = ImageClassifier(picsize, lst_dogs,
                                 out_channels = 12,
                                 out_channels_2 = 24,
                                 hidden_units = 50,
                                 regularization_strength = 0.5,
                                 batch_size = 50,
                                 learning_rate = 0.0001,
                                 convolution_size = 5,
                                 pool_size = 2,
                                 training_epochs = 100,
                                 loss_threshold = 0.01,
                                 verbose=True,
                                 grid_search=True)

    model.fit(Xtrain_run, Ytrain_run)
    score = model.score(Xtest, Ytest_1)
    print(score)


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
