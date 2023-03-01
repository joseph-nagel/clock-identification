# Clock identification

An old image classification case study is contained in this repository.
It merely serves backup and archival purposes at this point.
The goal was to assess the quality of customer-provided product photos
and to classify whether or not a clock is contained in such an image.
Unfortunately, the data cannot be shared here.

## Overview

The [first notebook](notebooks/1_quality_assessment.ipynb) starts with the initial
exploration of the data followed by a blurriness-based quality assessment.
An image classifier is trained in the [second notebook](notebooks/2_binary_classification.ipynb).
It is based on a pretrained feature extractor and transfer learning.
Finally, after starting a simple inference server by running `python app.py`,
the [last notebook](notebooks/3_final_models.ipynb) demonstrates how
the image quality and the classifier model could be deployed in practice.

