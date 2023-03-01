'''Data import.'''

import os

import numpy as np
from PIL import Image


class DataImport():
    '''
    Simple class to import the data.

    Parameters
    ----------
    dir_path : str or path object
        Path to directory from where to import the data.
    guess_label : None or callable
        Determines whether and how labels are guessed from filenames.

    '''

    def __init__(self, dir_path, guess_label=None):
        self.images, self.labels = self.load_dir(dir_path, guess_label)

    @staticmethod
    def load_dir(dir_path, guess_label=None):
        '''Load data from a directory.'''
        images = []

        if guess_label is not None:
            labels = []
        else:
            labels = None

        for subdir, dirs, files in os.walk(dir_path):
            file_names = sorted([f for f in files if f.endswith(('.jpg','.jpeg'))])

            for file_name in file_names:
                file_path = os.path.join(subdir, file_name)

                image = np.asarray(Image.open(file_path))
                images.append(image)

                if guess_label is not None:
                    label = guess_label(file_name)
                    labels.append(label)

        return images, labels

    def __len__(self):
        return len(self.images)

    def print_summary(self):
        '''Print a concise summary of the data.'''
        print('No. of images:', len(self))

        if self.labels is not None:
            for label in np.unique(self.labels):
                number = np.sum([l == label for l in self.labels])

                print('No. of {}s: {}'.format(label, number))

