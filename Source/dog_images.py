import os
from PIL import Image
from PIL import ImageOps
import numpy as np
from sklearn.model_selection import train_test_split


class DogImages(object):
    '''
    Class to manage files of images of Dogs, of different sizes and offsets,
    with the characters split into teset and training sets.
    '''

    def __init__(self, dognames, picsize, root='../Data', flatten=True):
        '''Create an object of the class.
        Parameters:
        ----------
        dognames: list of strings of dog names; these should match folder name
        root:      directory to store the image files
        '''
        self.dognames = dognames
        self.root = root
        self.image_id = 0
        self.imagesize = picsize
        self.counter = 0
        self.tt_split = 0.3
        if flatten:
            self.flatten = True
        else:
            self.flatten = False

    def _make_img(self, dimg, directory):
            width, height = dimg.size
            img = Image.new('RGB',
                            (self.imagesize, self.imagesize),
                            (255, 255, 255))
            offset = (int((self.imagesize - width) / 2),
                      int((self.imagesize - height) / 2))
            img.paste(dimg, offset)
            img.save(directory + '/image{}.png'.format(self.imageid))
            self.imageid += 1
            img = ImageOps.mirror(img)
            img.save(directory + '/image{}.png'.format(self.imageid))
            self.imageid += 1

    def _make_imgs(self, directory):
        '''
        '''
        # First, create directories if they don't exist
        for newdir in [directory +'/' + dogname for dogname in self.dognames]:
            if not os.path.exists(newdir):
                os.makedirs(newdir)
                os.makedirs(newdir + '/train')
                os.makedirs(newdir + '/test')

        for dogname in self.dognames:
            dogdir = directory + '/' + dogname
            for image in os.listdir("../Images/" + dogname):
                dimg = Image.open('../Images/' + dogname + '/' + image)
                w, h = dimg.size
                if w >= self.imagesize or h >= self.imagesize:
                    big = h
                    if w >= h:
                        big = w
                    img = Image.new('RGB',
                                    (big, big),
                                    (255, 255, 255))
                    offset = (int((big - w) / 2),
                              int((big - h) / 2))
                    img.paste(dimg, offset)
                    dimg = img.resize((self.imagesize,self.imagesize),
                                        Image.ANTIALIAS)
                if (self.train_cnt == 0 or self.test_cnt/self.train_cnt
                    >= self.tt_split):
                    self._make_img(dimg, dogdir + '/train')
                    self.train_cnt += 1
                else:
                    self._make_img(dimg, dogdir + '/test')
                    self.test_cnt += 1

    def generate_img_files(self):
        '''
        '''
        self.imageid = 0
        self.train_cnt = 0
        self.test_cnt = 0
        self._make_imgs(self.root)

    def _get_filenames(self, testtrain, dogname):
        '''Get full names of all image files to be loaded

        Parameters:
        ----------
        testtrain: either 'test' or 'train', depending on whence to load the file
        '''
        base = self.root + '/' + dogname + '/' + testtrain + '/'
        return [ base+directory for directory in os.listdir(base) ]

    def load_images(self, testtrain):
        '''Load the images files already created into arrays return two numpy
        arrays, one of shape (n, p), the other of shape (n, f), where n=number
        of data points, p=number of pixels, f=number of fonts Parameters:

        ----------

        testtrain: either 'test' or 'train', depending on whence to
        load the file
        '''
        imagedict = {}
        for dogname in self.dognames:
            imagedict[dogname] = self._get_filenames(testtrain, dogname)
        n = sum([len(imagedict[dogname]) for dogname in self.dognames])
        p = self.imagesize * self.imagesize * 3
        f = len(self.dognames)
        if self.flatten:
            ximages = np.zeros((n,p))
        else:
            ximages = np.zeros((n,self.imagesize,self.imagesize,3))
        yimages = np.zeros((n,f))
        i = 0
        for yi, dogname in enumerate(self.dognames):
            for imagename in imagedict[dogname]:
                if self.flatten:
                    ximages[i,:] = np.array(Image.open(imagename)).reshape((-1,))
                else:
                    ximages[i,:] = np.array(Image.open(imagename))
                yimages[i,yi] = 1
                i += 1
        return ximages, yimages
