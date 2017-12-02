from dog_images import DogImages
import os
import numpy as np
from keras import optimizers
from keras.layers import Input, Flatten, Dense
from keras.models import Sequential, Model, load_model
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import random
from keras.optimizers import SGD
import pickle
from keras.optimizers import Adam

def main():
    picsize = 224
    # Step 1: Get List of Dogs
    lst = [x[0] for x in os.walk('../Images')]
    lst_dogs = [a.replace('../Images/', '') for a in lst[1:]]

    dog_images = DogImages(lst_dogs, picsize, flatten=False)
    #dog_images.generate_img_files()
    train_imgs = dog_images.load_images('train')
    test_imgs = dog_images.load_images('test')
    Xtest = test_imgs[0]
    Ytest = test_imgs[1]
    Xtrain = train_imgs[0]
    Ytrain = train_imgs[1]

    lst = [x[0] for x in os.walk('../Images')]
    lst_dogs = [a[a.index('-')+1:] for a in lst[1:]]
    print(lst_dogs[0])

    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=True)#
    base_model = VGG16(weights='imagenet', include_top=True)

    # add a global spatial average pooling
    base_model.layers.pop()
    # x = base_model.output
    x = base_model.layers[-1].output
    # x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # x = Dense(1000, activation='relu')(x)
    # and a logistic layer -- let's say we have 120 classes
    predictions = Dense(120, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers[:len(model.layers[:])-1]:
        layer.trainable = False
    model.summary()


    model.compile(optimizer=SGD(), loss='categorical_crossentropy')
    bs = 32
    samp = int(Xtrain.shape[0]/bs)
    print(samp)
    best = 0.0
    N = 0
    while N<20 and best < 0.8:
        model.fit_generator(my_generator(Xtrain, Ytrain, bs),
                            samples_per_epoch=samp, nb_epoch=10, verbose=1)
        preds = model.predict(Xtest, verbose=1)
        new = np.sum((np.argmax(preds,1) == np.argmax(Ytest,1)))/Ytest.shape[0]
        if new > best:
            best = new
            print('SGD Best = {:.2f}%'.format(best*100.0))
            with open('SGD_version.pickle', 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ### ADAM
    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=True)#
    base_model = VGG16(weights='imagenet', include_top=True)

    # add a global spatial average pooling
    base_model.layers.pop()
    # x = base_model.output
    x = base_model.layers[-1].output
    # x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # x = Dense(1000, activation='relu')(x)
    # and a logistic layer -- let's say we have 120 classes
    predictions = Dense(120, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers[:len(model.layers[:])-1]:
        layer.trainable = False
    model.summary()


    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    bs = 32
    samp = int(Xtrain.shape[0]/bs)
    print(samp)
    best = 0.0
    N = 0
    while N<20 and best < 0.8:
        model.fit_generator(my_generator(Xtrain, Ytrain, bs),
                            samples_per_epoch=samp, nb_epoch=1, verbose=1)
        preds = model.predict(Xtest, verbose=1)
        new = np.sum((np.argmax(preds,1) == np.argmax(Ytest,1)))/Ytest.shape[0]
        if new > best:
            best = new
            print('ADAM Best = {:.2f}%'.format(best*100.0))
            with open('ADAM_version.pickle', 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def my_generator(X, y, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 224, 224, 3))
    batch_labels = np.zeros((batch_size,y.shape[1]))
    features = X.shape[0]
    indices = []
    while True:
        for i in range(batch_size):
            # choose random index in features
            try:
                index = indices[0]
                indices = indices[1:]
            except:
                indices = [a for a in range(X.shape[0] - 1)]
                random.shuffle(indices)
                index = indices[0]
                indices = indices[1:]
            batch_features[i] = X[index,:]
            batch_labels[i] = y[index,:]
        yield batch_features, batch_labels

if __name__ == '__main__':
    main()
