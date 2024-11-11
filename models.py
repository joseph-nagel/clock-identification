'''
Models for blur assessment and clock identification.

Summary
-------
Our final models for blur assessment and clock identification are provided.
They can be easily accessed through the class 'OLXModel'.
An instance of that class has a 'predict'-method that returns a blur score,
the probability of being a clock and the thresholded binary result.

'''

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from utils import BlurScore


# image preprocessing
SHAPE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


transform = {
    # resized images
    'resize': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    # cropped images
    'crop': transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    # full-sized
    'full': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
}


# binary model
pretrained_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

binary_model = nn.Sequential(
    pretrained_model.features,
    nn.AdaptiveAvgPool2d(output_size=(6, 6)),
    nn.Flatten(),
    nn.Linear(in_features=256*6*6, out_features=1)
)

binary_model.eval()


class OLXModel:
    '''
    Final model class that predicts bluriness and clockness.

    Parameters
    ----------
    weights_path : str or path object
        Path to a file with trained model weights.
    max_blur : int or float
        Maximum blurriness that will be normalized to one.

    '''

    def __init__(self, weights_path='weights.pt', max_blur=98.52080247322063):

        self.blur_score = BlurScore(max_blur=max_blur)

        self.binary_model = binary_model
        self.binary_model[-1].load_state_dict(torch.load(weights_path))

    def __call__(self, image, transform_mode='full', threshold=0.5):

        return self.predict(
            image,
            transform_mode=transform_mode,
            threshold=threshold
        )

    def predict(self, image, transform_mode='full', threshold=0.5):
        '''
        Predict bluriness and clockness.

        Parameters
        ----------
        image : array-like
            Array containing the image.
        transform_mode : {'full', 'resize', 'crop'}
            Determines how images are preprocessed.
        threshold : float in [0,1]
            Classification threshold parameter.

        '''

        array = np.asarray(image)
        tensor = transform[transform_mode](array)[None,...]

        blurriness = self.blur_score(array)

        with torch.no_grad():
            clockness = torch.sigmoid(self.binary_model(tensor)).item()

        is_clock = int(clockness >= threshold)

        return blurriness, clockness, is_clock

