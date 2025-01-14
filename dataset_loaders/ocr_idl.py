import io
import os
import random

import torch

from PIL import Image
from datasets import load_dataset

def format_data(sample, processor, config, use_ocr):
    images = []
    images_boxes = []
    text = []
    text_boxes = []    
    
    prev_document = ''
    for document, page, ocr_tokens, ocr_boxes, image in zip(sample['images_id'], sample['doc_pages'], sample['ocr_tokens'], sample['ocr_boxes'], sample['images']):
        current_document = document[0].split('_')[0]
        document_id = random.randint(1, 1000) if prev_document != current_document else document_id
        prev_document = current_document
        page = page if page < 1000 else 999

        image = Image.open(io.BytesIO(image[0]))
        images.append(image)
        images_boxes.append([document_id, page])

        text.extend(ocr_tokens[0])
        ocr_boxes = [[document_id, page] + box for box in ocr_boxes]
        text_boxes.extend(ocr_boxes)

    input_ids = torch.tensor([])
    input_boxes = torch.tensor([])
    for word, box in zip(text, text_boxes):
        word_ids = torch.tensor(processor.tokenizer.encode(word, add_special_tokens=False))
        input_ids = torch.cat([input_ids, word_ids])
        input_boxes = torch.cat([input_boxes, torch.tensor(box).repeat(len(word_ids), 1)])
        
    input_ids = input_ids[:processor.tokenizer.model_max_length]
    input_boxes = input_boxes[:processor.tokenizer.model_max_length]

    mask_spans = []
    i = 1
    while i < len(input_ids) - 1:
        if len(mask_spans) < 100 and random.random() < config['mask_probability']:
            start = i
            end = i + random.randint(1, 5)  # create a span of 1， 2 or 3 or 4， 5.
            end = min(end, len(input_ids) - 2)
            mask_spans.append([start, end])
            i = end + 2
        else:
            i += 1
    
    mask_ID_counter = 0
    labels = torch.tensor([])
    box_labels = torch.tensor([])
    new_input_ids = torch.tensor([])
    new_input_boxes = torch.tensor([])
    previous_end = 0

    for start, end in mask_spans:
        extra_id = torch.tensor([processor.tokenizer.convert_tokens_to_ids(f"<extra_id_{mask_ID_counter}>")])
        labels = torch.cat([labels, extra_id, input_ids[start:end+1]])
        new_input_ids = torch.cat([new_input_ids, input_ids[previous_end:start], extra_id])
        new_input_boxes = torch.cat([new_input_boxes, input_boxes[previous_end:start],
                                    torch.tensor([[input_boxes[start:end+1][0, 0], input_boxes[start:end+1][0, 1],
                                                torch.min(input_boxes[start:end+1][:, 2]), torch.min(input_boxes[start:end+1][:, 3]),
                                                torch.max(input_boxes[start:end+1][:, 4]), torch.max(input_boxes[start:end+1][:, 5])]])])

        box_labels = torch.cat([box_labels, 
                                torch.tensor([[input_boxes[previous_end:start][0, 0], input_boxes[previous_end:start][0, 1],
                                            torch.min(input_boxes[previous_end:start][:, 2]), torch.min(input_boxes[previous_end:start][:, 3]),
                                            torch.max(input_boxes[previous_end:start][:, 4]), torch.max(input_boxes[previous_end:start][:, 5])]]), input_boxes[start:end+1]])
        

        
        previous_end = end + 1
        mask_ID_counter += 1
    
    new_input_ids = torch.cat([new_input_ids, input_ids[previous_end:]])
    new_input_boxes = torch.cat([new_input_boxes, input_boxes[previous_end:]])

    if not use_ocr:
        # Remove a config['random_token_removal'] percentage of tokens not including the special tokens
        index = torch.tensor([x not in processor.tokenizer.all_special_ids for x in new_input_ids])
        non_special_tokens = new_input_ids[index]
        special_tokens = new_input_ids[~index]
        num_tokens = len(non_special_tokens)
        num_tokens_to_remove = int(num_tokens * config['random_token_removal'])
        random_indices = torch.randperm(num_tokens)[:num_tokens_to_remove]
        new_input_ids = torch.cat([special_tokens, non_special_tokens[random_indices]])

    sample['images'] = [images]
    sample['images_boxes'] = [images_boxes]
    sample['input_ids'] = new_input_ids.unsqueeze(0)
    sample['input_boxes'] = new_input_boxes.unsqueeze(0)
    sample['labels'] = labels.unsqueeze(0)

    return sample

def process(batch, processor):
    return processor(**batch)

def build_ocr_idl(config, split, processor):
    assert split == 'train', 'Only train split is available for OCR-IDL dataset'
    
    config['random_token_removal'] = config.get('random_token_removal', 0.80)
    config['mask_probability'] = config.get('mask_probability', 0.15)
    
    data_files = {"train": "train-*.parquet"}
    dataset = load_dataset(os.path.join(config['data_dir'], 'OCR-IDL', 'data'), data_files=data_files, split=split, streaming=True)

    dataset = dataset.filter(lambda x: len(x['ocr_tokens'][0]) > 50) # Filter out samples with less than 50 tokens
    dataset = dataset.map(format_data, batched=True, batch_size=config.get('num_pages', 1), fn_kwargs={'processor': processor, "config": config, "use_ocr": 'ocr' in config['model']}, remove_columns=['ocr_tokens', 'ocr_boxes', 'images_id', 'doc_pages', 'metadata'])
    #dataset = dataset.map(process, fn_kwargs={'processor': processor}, batched=True, batch_size=config['batch_size'], drop_last_batch=True, remove_columns=['images_boxes', 'input_boxes'])
    
    return dataset