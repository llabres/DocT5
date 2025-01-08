"""
Processor class for DocT5.
"""
import torch

from typing import List, Optional, Union

import transformers
from transformers import AutoImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from transformers.utils import logging

from image_processing_doct5 import DocT5ImageProcessor

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
        patches = self.image_processor.preprocess(images, boxes, **kwargs)['flattened_patches']
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

        if batch_images is not None:
            batch_processed_images = []
            batch_images_boxes = []
            batch_images = [[images] if type(images) != list else images for images in batch_images]
            images_boxes = [[[1, 1]]*len(images) for images in batch_images] if images_boxes is None else images_boxes
            for images in batch_images:
                images, boxes = self._process_images(images, images_boxes, **kwargs)
                batch_processed_images.append(images)
                batch_images_boxes.append(boxes)
            padding_size = max([images.size(0) for images in batch_processed_images])
            batch_images = torch.stack([torch.nn.functional.pad(images, (0, padding_size - images.size(0)), 'constant', value=0) for images in batch_processed_images])
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
        return ["input_ids", "images", "boxes", "attention_mask"]