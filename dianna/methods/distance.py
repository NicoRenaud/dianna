import os
import random
from urllib.parse import urlparse
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from matplotlib import pyplot as plt
from requests import get
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
from dianna.methods.rise import generate_masks_for_images
from dianna import utils
from sklearn.metrics import pairwise_distances


class DistanceExplainer:
    # axis labels required to be present in input image data
    required_labels = ('channels', )

    def __init__(self, n_masks=1000, feature_res=8, p_keep=.5,  # pylint: disable=too-many-arguments
                 p_keep_lowest_distances=.2, axis_labels=None, batch_size=10, preprocess_function=None):
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.axis_labels = axis_labels if axis_labels is not None else []
        self.p_keep_lowest_distances = p_keep_lowest_distances
        self.batch_size = batch_size

    def explain_image_distance(self, model_or_function, input_data, embedded_reference, **explain_distance_kwargs):
        input_data = utils.to_xarray(input_data, self.axis_labels, DistanceExplainer.required_labels)
        # add batch axis as first axis
        input_data = input_data.expand_dims('batch', 0)
        input_data, full_preprocess_function = self._prepare_image_data(input_data)
        runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)
        
        active_p_keep = 0.5 if self.p_keep is None else self.p_keep  # Could autotune here (See #319)

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        # Expose masks for to make user inspection possible
        self.masks = generate_masks_for_images(img_shape, active_p_keep, self.n_masks, self.feature_res)
        
        # Make sure multiplication is being done for correct axes
        masked = input_data * self.masks

        batch_predictions = []
        for i in tqdm(range(0, self.n_masks, self.batch_size), desc='Explaining'):
            batch_predictions.append(runner(masked[i:i + self.batch_size]))
        self.predictions = np.concatenate(batch_predictions)

        reference_pred = embedded_reference
        distances = pairwise_distances(self.predictions, reference_pred, metric='cosine') / 2
        lowest_distances_indices = np.argsort(distances, axis=0)[:int(len(self.predictions) * self.keep_lowest_distance_masks_fraction)]

        mask_weights = np.exp(-distances[lowest_distances_indices])
        lowest_distances_masks = self.masks[lowest_distances_indices]

        sal = mask_weights.T.dot(lowest_distances_masks.reshape(len(lowest_distances_masks), -1)).reshape(-1, *img_shape)

        normalization = mask_weights.sum()

        sal = sal / normalization
        return sal

    
    def _prepare_image_data(self, input_data):
        """Transforms the data to be of the shape and type RISE expects.

        Args:
            input_data (xarray): Data to be explained

        Returns:
            transformed input data, preprocessing function to use with utils.get_function()
        """
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = utils.move_axis(input_data, 'channels', -1)
        # create preprocessing function that puts model input generated by RISE into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(channels_axis_index, input_data.dtype)
        return input_data, full_preprocess_function

    def _get_full_preprocess_function(self, channel_axis_index, dtype):
        """Creates a full preprocessing function.

        Creates a preprocessing function that incorporates both the (optional) user's
        preprocessing function, as well as any needed dtype and shape conversions

        Args:
            channel_axis_index (int): Axis index of the channels in the input data
            dtype (type): Data type of the input data (e.g. np.float32)

        Returns:
            Function that first ensures the data has the same shape and type as the input data,
            then runs the users' preprocessing function
        """
        def moveaxis_function(data):
            return utils.move_axis(data, 'channels', channel_axis_index).astype(dtype).values

        if self.preprocess_function is None:
            return moveaxis_function
        return lambda data: self.preprocess_function(moveaxis_function(data))