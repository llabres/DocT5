"""
Processor class for DocT5.
"""
import torch

import math
import numpy as np
from typing import List, Optional, Union, Dict

import transformers
from transformers import AutoImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import convert_to_rgb, normalize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType, is_torch_available, is_vision_available, logging
from transformers.utils.import_utils import requires_backends

logger = logging.get_logger(__name__)

# adapted from: https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    requires_backends(torch_extract_patches, ["torch"])

    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

class DocT5ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DocT5 image processor. Adapted from the Pix2Struct image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
            deviation.
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image.
        max_patches (`int`, *optional*, defaults to 2048):
            The maximum number of patches to extract from the image as per the [Pix2Struct
            paper](https://arxiv.org/pdf/2210.03347.pdf).
    """

    model_input_names = ["flattened_patches"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_normalize: bool = True,
        patch_size: Dict[str, int] = None,
        max_patches: int = 2048,
        random_patch_removal: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.max_patches = max_patches
        self.random_patch_removal = random_patch_removal

    def extract_flattened_patches(
        self,
        image: np.ndarray,
        max_patches: int,
        patch_size: dict,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.

        Returns:
            result (`np.ndarray`):
                A sequence of `max_patches` flattened patches.
        """
        requires_backends(self.extract_flattened_patches, "torch")

        # convert to torch
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
        image = torch.from_numpy(image)

        patch_height, patch_width = patch_size["height"], patch_size["width"]
        image_height, image_width = get_image_size(image, ChannelDimension.FIRST)

        # maximize scale s.t.
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        # [1, rows, columns, patch_height * patch_width * image_channels]
        patches = torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # Offset by 1 so the ids do not contain zeros, which represent padding.
        row_ids += 1
        col_ids += 1

        # Prepare additional patch features.
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        result = to_numpy_array(result)

        return result

    def normalize(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        The image std is to mimic the tensorflow implementation of the `per_image_standardization`:
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # take mean across the whole `image`
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        return normalize(
            image,
            mean=mean,
            std=adjusted_stddev,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        boxes: list = None,
        do_convert_rgb: bool = None,
        do_normalize: Optional[bool] = None,
        max_patches: Optional[int] = None,
        patch_size: Optional[Dict[str, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> ImageInput:
        """
        Preprocess an image or batch of images. The processor first computes the maximum possible number of
        aspect-ratio preserving patches of size `patch_size` that can be extracted from the image. It then pads the
        image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the
        images are standardized following the tensorflow implementation of `per_image_standardization`
        (https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization).


        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            max_patches (`int`, *optional*, defaults to `self.max_patches`):
                Maximum number of patches to extract.
            patch_size (`dict`, *optional*, defaults to `self.patch_size`):
                Dictionary containing the patch height and width.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_patches = max_patches if max_patches is not None else self.max_patches

        if kwargs.get("data_format", None) is not None:
            raise ValueError("data_format is not an accepted input as the outputs are ")

        images = make_list_of_images(images) if not isinstance(images, list) else images

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_normalize:
            images = [self.normalize(image=image, input_data_format=input_data_format) for image in images]

        # convert to torch tensor and permute
        image_patches = []
        for image, box in zip(images, boxes):
            image_max_patches = min(max_patches, (image.shape[0]//patch_size["height"])*(image.shape[1]//patch_size["width"]))
            image = self.extract_flattened_patches(
                image=image, max_patches=image_max_patches, patch_size=patch_size, input_data_format=input_data_format
            )
            box = np.array([box]*image_max_patches)
            n_cols = image[:, 1].max()
            n_rows = image[:, 0].max()
            image = np.concatenate([box,
                                    ((image[:, 0] - 1) / n_rows).reshape(-1, 1),
                                    ((image[:, 1] - 1) / n_cols).reshape(-1, 1),
                                    (image[:, 0] / n_rows).reshape(-1, 1),
                                    (image[:, 1] / n_cols).reshape(-1, 1),
                                    image[:, 2:]], axis=1)
            image_patches.append(image)
        
        encoded_outputs = BatchFeature(
            data={"flattened_patches": image_patches}, tensor_type=return_tensors
        )

        return encoded_outputs

AutoImageProcessor.register("DocT5ImageProcessor", DocT5ImageProcessor)
transformers.DocT5ImageProcessor = DocT5ImageProcessor

class DocT5ImagesKwargs(ImagesKwargs, total=False):
    max_patches: Optional[int]

class DocT5Processor(ProcessorMixin):
    r"""
    Constructs a DocT5 processor which wraps a T5 tokenizer and DocT5 image processor into a single
    processor.

    [`DocT5Processor`] offers all the functionalities of [`DocT5ImageProcessor`] and [`T5TokenizerFast`]. See
    the docstring of [`~DocT5Processor.__call__`] and [`~DocT5Processor.decode`] for more information.

    Args:
        image_processor (`DocT5ImageProcessor`):
            An instance of [`DocT5ImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "DocT5ImageProcessor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)
        self.patch_size = image_processor.patch_size

    def _remove_patches(self, patches):
        patches = patches[(torch.std(patches[:, 6:][:, 0::3], dim=1) > 0.1) + (torch.std(patches[:, 6:][:, 1::3], dim=1) > 0.1) + (torch.std(patches[:, 6:][:, 2::3], dim=1) > 0.1)] # Remove patches with low variance in all three channels
        patches = patches[patches[:, 4] != 0] # Remove padding patches if any

        if self.image_processor.random_patch_removal > 0:
            # remove random_patch_removal% of patches at random
            num_patches = patches.shape[0]
            num_patches_to_remove = int(self.image_processor.random_patch_removal*num_patches)
            patches = patches[torch.randperm(num_patches)[:-num_patches_to_remove]]

        return patches


    def _process_images(self, images, boxes, **kwargs):
        patches = self.image_processor.preprocess(images, boxes, return_tensors='pt')['flattened_patches']
        patches = self._remove_patches(patches.view(-1, patches.size(-1)))
        boxes = patches[:, :6]
        patches = patches[:, 6:]

        return patches, boxes      


    def _process_text(self, text, boxes, **kwargs):
        input_ids = torch.tensor([])
        input_boxes = torch.tensor([])
        for word, box in zip(text, boxes):
            word_ids = torch.tensor(self.tokenizer.encode(word, add_special_tokens=False))
            input_ids = torch.cat([input_ids, word_ids])
            input_boxes = torch.cat([input_boxes, torch.tensor(box).repeat(len(word_ids), 1)])

        return input_ids[:self.tokenizer.model_max_length], input_boxes[:self.tokenizer.model_max_length]

    def __call__(
        self,
        images=None,
        text=None,
        text_boxes=None,
        images_boxes=None,
        labels=None,
        input_ids=None,
        input_boxes=None,
        audio=None,
        videos=None,
        **kwargs: Unpack[DocT5ImagesKwargs],
    ) -> Union[BatchEncoding, BatchFeature]:
        """
        This method uses [`DocT5ImageProcessor.preprocess`] method to prepare image(s) for the model, and
        [`T5TokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")            

        batch_images = images if type(images) == list or images is None else [images]
        batch_texts = text if type(text) == list or text is None else [text]

        batch_input_ids = input_ids
        batch_input_boxes = input_boxes

        if batch_images is not None:
            batch_processed_images = []
            batch_images_boxes = []
            batch_images = [[images] if type(images) != list else images for images in batch_images]
            images_boxes = [[[1, 1]]*len(images) for images in batch_images] if images_boxes is None else images_boxes
            for images, boxes in zip(batch_images, images_boxes):
                images, boxes = self._process_images(images, boxes, **kwargs)
                batch_processed_images.append(images)
                batch_images_boxes.append(boxes)
            padding_size = max([images.size(0) for images in batch_processed_images])
            batch_images = torch.stack([torch.nn.functional.pad(images, (0, 0, 0, padding_size - images.size(0)), 'constant', value=0) for images in batch_processed_images])
            batch_images_boxes = torch.stack([torch.nn.functional.pad(boxes, (0, 0, 0, padding_size - boxes.size(0)), 'constant', value=0) for boxes in batch_images_boxes])
            visual_attention_mask = batch_images_boxes[:, :, 0] != 0
        else:
            batch_images = None
            batch_images_boxes = torch.tensor([])
            visual_attention_mask = torch.tensor([])

        if batch_texts is not None:
            batch_input_ids = []
            batch_input_boxes = []
            batch_texts = [text.split(" ") if isinstance(text, str) else text for text in batch_texts]
            batch_boxes = [[[0,0,0,0,0,0]]*len(text) for text in batch_texts] if text_boxes is None else text_boxes
            for text, boxes in zip(batch_texts, batch_boxes):
                input_ids, input_boxes = self._process_text(text, boxes, **kwargs)
                batch_input_ids.append(input_ids)
                batch_input_boxes.append(input_boxes)

        if batch_input_ids is not None:
            padding_size = max([len(input_ids) for input_ids in batch_input_ids])
            batch_input_ids = torch.stack([torch.nn.functional.pad(input_ids, (0, padding_size - len(input_ids)), 'constant', value=self.tokenizer.pad_token_id) for input_ids in batch_input_ids])
            batch_input_boxes = torch.stack([torch.nn.functional.pad(input_boxes, (0, 0, 0, padding_size - input_boxes.size(-2)), 'constant', value=0) for input_boxes in batch_input_boxes])
            attention_mask = batch_input_ids != self.tokenizer.pad_token_id 
        else:
            batch_input_ids = None
            batch_input_boxes = torch.tensor([])
            attention_mask = torch.tensor([])
        
        boxes = torch.cat((batch_input_boxes, batch_images_boxes), dim=1)
        boxes[:, :, 2:] = boxes[:, :, 2:] * 1000
        attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=1)

        if labels is not None:
            labels = labels if type(labels) == list else [labels]
            if type(labels[0]) == torch.Tensor:
                padding_size = max([len(label) for label in labels])
                labels = torch.stack([torch.nn.functional.pad(label, (0, padding_size - len(label)), 'constant', value=self.tokenizer.pad_token_id) for label in labels]).to(torch.int64)
                decoder_attention_mask = (labels != self.tokenizer.pad_token_id).to(torch.int64)
            else:
                labels = self.tokenizer(labels, padding='longest', return_tensors='pt', add_special_tokens=True, truncation=True)
                decoder_attention_mask = labels['attention_mask']
                labels = labels['input_ids']
        else:
            labels = None
            decoder_attention_mask = None
            
        return dict(
            input_ids=batch_input_ids.to(torch.int32),
            images=batch_images.to(torch.float32),
            boxes=boxes.to(torch.int32),
            attention_mask=attention_mask.to(torch.int32),
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    
    @property
    def model_input_names(self):
        return ["input_ids", "images", "boxes"]