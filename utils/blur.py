'''Blur detection.'''

import numpy as np
from skimage.filters import laplace
from skimage.color import rgb2gray


class BlurScore():
    '''
    Blurriness score based that can be normalized by the max. over an image set.

    Summary
    -------
    This class allows for computing a blurriness score for images.
    It can be divided by the max. unnormalized score over a set of images.
    The resulting normalized score is (0,1]-valued over the original images.
    When applied to new images though, the score might well be higher than 1.

    Parameters
    ----------
    max_blur : int or float
        Maximum blurriness that will be normalized to one.

    Notes
    -----
    Note that the "fit", "evaluate", "fit_evaluate" methods resemble
    the fit/transform-interface of scikit-learn image transformations.

    '''

    def __init__(self, max_blur=1):
        self.max_blur = max_blur

    def __call__(self, images, normalize=True):
        return self.evaluate(images, normalize)

    def fit(self, images):
        '''Determine the max. blurriness and return the class instance.'''
        blurriness = self.evaluate(images, normalize=False)
        self.max_blur = np.max(blurriness)
        return self

    def fit_evaluate(self, images):
        '''Determine the max. blurriness and return normalized scores.'''
        blurriness = self.evaluate(images, normalize=False)
        self.max_blur = np.max(blurriness)
        return blurriness / self.max_blur

    def evaluate(self, images, normalize=True):
        '''Compute the (normalized) blurriness scores.'''
        # images in list
        if isinstance(images, list):
            blurriness = np.array([blur_score(image) for image in images])
        # image(s) as array
        elif isinstance(images, np.ndarray):
            if images.ndim == 4: # array with first axis as image id
                blurriness = np.array([blur_score(image) for image in images])
            elif images.ndim in (2, 3):
                blurriness = blur_score(images)
            else:
                raise ValueError('Image has the wrong shape')
        else:
            raise ValueError('Image has the wrong type')

        # normalization
        if normalize:
            blurriness /= self.max_blur

        return blurriness


def blur_score(image, small_eps=1e-06):
    '''
    Compute unnormalized blurriness score based on the std. of the Laplacian.

    Summary
    -------
    The blurriness of an image is here defined as the reciprocal of the standard
    deviation of the response of the grayscaled image under a Laplacian filter.
    Since the latter is often used as an edge detector, this definition
    formalizes an intuitive notion of blurriness as a lack of sharp edges.

    '''

    image = np.asarray(image)

    if image.ndim in (2, 3):
        sharpness = laplace(rgb2gray(image)).std() # std. instead of variance
        blurriness = 1 / (sharpness + small_eps) # avoid division by zero
        return blurriness
    else:
        raise ValueError('Image has the wrong shape')

