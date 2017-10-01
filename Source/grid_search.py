import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from dog_images import DogImages
from image_classifier import ImageClassifier
from sklearn.model_selection import GridSearchCV

def main():
    picsize = 200
    # Step 1: Get List of Dogs
    lst = [x[0] for x in os.walk('../Images')]
    lst_dogs = [a.replace('../Images\\', '') for a in lst[1:]]
    print(lst_dogs)

    # Step 2: Make the Images...
    dog_images = DogImages(lst_dogs, picsize)
    dog_images.generate_img_files()
    train_imgs = dog_images.load_images('train')
    test_imgs = dog_images.load_images('test')
    Xtest = test_imgs[0]
    Ytest = test_imgs[1]
    Xtrain = train_imgs[0]
    Ytrain = train_imgs[1]

    # Step 3: Initial Shuffle of Train & Test Sets
    Xhold = Xtrain.copy()
    Yhold = Ytrain.copy()
    new = np.array([i for i in range(Xhold.shape[0])])
    np.random.shuffle(new)
    for i, n in enumerate(new):
        Xtrain[i, :] = Xhold[n, :]
        Ytrain[i, :] = Yhold[n, :]

    Xhold = Xtest
    Yhold = Ytest
    new = np.array([i for i in range(Xhold.shape[0])])
    np.random.shuffle(new)
    for i, n in enumerate(new):
        Xtest[i, :] = Xhold[n, :]
        Ytest[i, :] = Yhold[n, :]

    # Step 4: Grid Search...
    params = {'picsize':[picsize],
              'classes':[lst_dogs],
              'out_channels':[75, 150, 225, 300, 375],
              'out_channels_2':[150, 300, 450, 600],
              'hidden_units':[150, 225, 300, 375],
              'regularization_strength':[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
              'batch_size':[32, 64, 96, 100, 150, 200],
              'learning_rate':[0.00001, 0.0001, 0.001, 0.01],
              'loss_threshold':[0.5],
              'verbose':[True]}

    gs = GridSearchCV(ImageClassifier(), params)
    gs.fit(Xtrain, Ytrain)
    print()
    print('Best Accuracy: {:.3f}'.format(gs.best_score_))
    print('Best Params: {}'.format(gs.best_params_))

if __name__ == '__main__':
    main()
