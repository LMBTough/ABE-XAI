import matplotlib.pyplot as plt
from trulens_eval.visualizations import Visualizer, get_backend
import matplotlib.pyplot as plt
import numpy as np
import os
from abe.func.utils import _check_device
from abe.type import ModelType
import torch
import cv2
from torchvision.ops import nms

os.environ['TRULENS_BACKEND'] = 'pytorch'


class HeatmapVisualizer(Visualizer):
    """
    Visualizes attributions by overlaying an attribution heatmap over the
    original image, similar to how GradCAM visualizes attributions.
    """

    def __init__(
        self,
        overlay_opacity=0.5,
        normalization_type=None,
        blur=10.,
        cmap='jet'
    ):
        super().__init__(
            combine_channels=True,
            normalization_type=normalization_type,
            blur=blur,
            cmap=cmap
        )

        self.default_overlay_opacity = overlay_opacity

    def __call__(
        self,
        attributions,
        x,
        output_file=None,
        imshow=True,
        fig=None,
        return_tiled=False,
        overlay_opacity=None,
        normalization_type=None,
        blur=None,
        cmap=None
    ) -> np.ndarray:
        _, normalization_type, blur, cmap = self._check_args(
            attributions, None, normalization_type, blur, cmap
        )

        # Combine the channels.
        attributions = attributions.mean(
            axis=get_backend().channel_axis, keepdims=True
        )

        # Blur the attributions so the explanation is smoother.
        if blur:
            attributions = self._blur(attributions, blur)

        # Normalize the attributions.
        attributions = self._normalize(attributions, normalization_type)

        tiled_attributions = self.tiler.tile(attributions)

        # Normalize the pixels to be in the range [0, 1].
        x = self._normalize(x, '01')
        tiled_x = self.tiler.tile(x)

        if cmap is None:
            cmap = self.default_cmap

        if overlay_opacity is None:
            overlay_opacity = self.default_overlay_opacity

        # Display the figure:
        _fig = plt.figure() if fig is None else fig

        plt.axis('off')
        plt.imshow(tiled_x)
        plt.imshow(tiled_attributions, alpha=overlay_opacity, cmap=cmap)

        if output_file:
            plt.savefig(output_file, bbox_inches=0)

        if imshow:
            plt.show()

        elif fig is None:
            plt.close(_fig)

        return tiled_x, tiled_attributions if return_tiled else attributions


G = [0, 255, 0]
R = [255, 0, 0]


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70, low=0.2, plot_distribution=False):  # 99.9 70.0
    m = compute_threshold_by_top_percentage(
        attributions, percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(
        attributions, percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (transformed >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError
    return threshold


def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def visualize(attributions, image, positive_channel=G, negative_channel=R, polarity='positive',
              clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False,
              structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True,
              mask_mode=False, plot_distribution=False):
    if polarity == 'both':
        raise NotImplementedError

    elif polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
        channel = positive_channel

    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(attributions, clip_above_percentile,
                                    clip_below_percentile, 0.0, plot_distribution=plot_distribution)
    attributions = np.power(attributions, 0.75)
    attributions_mask = attributions.copy()
    if morphological_cleanup:
        raise NotImplementedError
    if outlines:
        raise NotImplementedError
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 2)
            attributions = np.clip(attributions * image, 0, 255)
            attributions = attributions[:, :, (2, 1, 0)]
    return attributions


device = _check_device()
mask_viz = HeatmapVisualizer(blur=7, normalization_type="signed_max")


def plot_explanation_heatmap(attribution, input):
    input_dim = input.dim()
    attribution_dim = len(attribution.shape)
    if input_dim == 3 and attribution_dim == 3:
        input = input.unsqueeze(0)
        attribution = np.expand_dims(attribution, axis=0)
    assert input.dim() == 4 and len(attribution.shape) == 4
    im_, mask = mask_viz(attribution, input.cpu().detach(
    ).numpy(), overlay_opacity=0.5, imshow=False, return_tiled=True)
    plt.figure()
    plt.imshow(im_)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.show()
    
    
    
def plot_adversarial_sample(forward_fn,batch,adversarial_sample, model_type):
    match model_type:
        case ModelType.IMAGECLASSIFICATION:
            input, label,*extra = batch
            pred = forward_fn(batch).argmax(dim=-1).item()
            adv_pred = forward_fn([adversarial_sample, label, *extra]).argmax(dim=-1).item()
            plt.figure()
            fig,axes = plt.subplots(1,2)
            axes[0].imshow(input[0].cpu().permute(1,2,0))
            axes[0].set_title(f'Original: {pred}')
            axes[1].imshow(adversarial_sample[0].cpu().permute(1,2,0))
            axes[1].set_title(f'Adversarial: {adv_pred}')
            plt.show()
        case ModelType.OBJECTDETECTION:
            input, target,*extra = batch
            plt.figure()
            fig,axes = plt.subplots(1,2)
            ori_boxes, ori_labels = forward_fn(batch)
            adv_boxes, adv_labels = forward_fn([adversarial_sample, target, *extra])
            ori_img = (input[0].cpu().permute(1,2,0) * 255).numpy().astype(np.uint8)
            adv_img = (adversarial_sample[0].cpu().permute(1,2,0) * 255).numpy().astype(np.uint8)
            ori_img = np.ascontiguousarray(ori_img)
            adv_img = np.ascontiguousarray(adv_img)

            for box, label in zip(ori_boxes, ori_labels):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ori_img = cv2.putText(ori_img, str(label.item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for box, label in zip(adv_boxes, adv_labels):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                adv_img = cv2.rectangle(adv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                adv_img = cv2.putText(adv_img, str(label.item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            axes[0].imshow(ori_img)
            axes[1].imshow(adv_img)
            plt.show()
            
        case _:
            raise NotImplementedError