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
    required_labels = ('channels',)

    def __init__(self, n_masks=1000, feature_res=8, p_keep=.5,  # pylint: disable=too-many-arguments
                 mask_selection_range_max=0.2, mask_selection_range_min=0, mask_selection_negative_range_max=1,
                 mask_selection_negative_range_min=0.8, axis_labels=None, batch_size=10,
                 preprocess_function=None):
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.axis_labels = axis_labels if axis_labels is not None else []
        self.mask_selection_range_max = mask_selection_range_max
        self.mask_selection_range_min = mask_selection_range_min
        self.mask_selection_negative_range_max = mask_selection_negative_range_max
        self.mask_selection_negative_range_min = mask_selection_negative_range_min
        self.batch_size = batch_size

    def explain_image_distance(self, model_or_function, input_data, embedded_reference, **explain_distance_kwargs):
        """

        :param model_or_function:
        :param input_data:
        :param embedded_reference:
        :param explain_distance_kwargs:
        :return: saliency map and the neutral value within the saliency map which indicates the parts of the image that
        neither bring the image closer nor further away from the embedded reference.
        """
        full_preprocess_function, input_data = self._prepare_input_data(input_data)
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
            new_predictions = runner(masked[i:i + self.batch_size])
            batch_predictions.append(new_predictions)

        self.predictions = np.concatenate(batch_predictions)

        lowest_distances_masks, lowest_mask_weights = self._get_lowest_distance_masks_and_weights(embedded_reference,
                                                                                           self.predictions, self.masks,
                                                                                           self.mask_selection_range_min,
                                                                                           self.mask_selection_range_max)
        highest_distances_masks, highest_mask_weights = self._get_lowest_distance_masks_and_weights(embedded_reference,
                                                                                           self.predictions, self.masks,
                                                                                           self.mask_selection_negative_range_min,
                                                                                           self.mask_selection_negative_range_max)

        def describe(x, name):
            print('Description of ', name)
            print('mean:', np.mean(x))
            print('std:', np.std(x))
            print('min:', np.min(x))
            print('max:', np.max(x))
        describe(highest_mask_weights, 'highest_mask_weights')
        describe(lowest_mask_weights, 'lowest_mask_weights')

        unnormalized_sal_lowest = np.mean(lowest_distances_masks, axis=0)
        unnormalized_sal_highest = np.mean(highest_distances_masks, axis=0)
        unnormalized_sal = unnormalized_sal_lowest - unnormalized_sal_highest

        saliency = unnormalized_sal

        input_prediction = runner(input_data)
        input_distance = pairwise_distances(input_prediction, embedded_reference, metric='cosine') / 2
        neutral_value = np.exp(-input_distance)

        return saliency, neutral_value

    @staticmethod
    def _get_lowest_distance_masks_and_weights(embedded_reference, predictions, masks, mask_selection_range_min,
                                               mask_selection_range_max):
        distances = pairwise_distances(predictions, embedded_reference,
                                       metric='cosine') / 2  # divide by 2 to have [0.1] output range
        lowest_distances_indices = np.argsort(distances, axis=0)[
                                   int(len(predictions) * mask_selection_range_min)
                                   :int(len(predictions) * mask_selection_range_max)]
        mask_weights = np.exp(-distances[lowest_distances_indices])
        lowest_distances_masks = masks[lowest_distances_indices]
        return lowest_distances_masks, mask_weights

    def _prepare_input_data(self, input_data):
        input_data_xarray = utils.to_xarray(input_data, self.axis_labels, DistanceExplainer.required_labels)
        input_data_xarray_expanded = input_data_xarray.expand_dims('batch', 0)
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data_xarray_expanded.dims.index('channels')
        prepared_input_data = utils.move_axis(input_data_xarray_expanded, 'channels', -1)
        # create preprocessing function that puts model input generated by RISE into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(channels_axis_index, prepared_input_data.dtype)
        return full_preprocess_function, prepared_input_data

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
