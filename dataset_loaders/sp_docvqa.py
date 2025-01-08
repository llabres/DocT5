import os
import io
import random
from PIL import Image
from datasets import load_dataset

def format_data(sample, use_ocr):
    idx = random.randint(0, len(sample['questions'])-1)

    document_id = random.randint(1, 1000)
    page_id = int(sample['page'])
    images = []
    images_boxes = []
    ocr_tokens = []
    ocr_boxes = []

    question = sample['questions'][idx]['question'].split(" ") 
    ocr_tokens += question
    ocr_boxes  += [[0, 0, 0, 0, 0, 0]]*len(question)

    for image in sample['images']:
        image = Image.open(io.BytesIO(image))
        images.append(image)
        images_boxes.append([document_id, page_id])

    if use_ocr:
        ocr_tokens.extend(sample['ocr_tokens'][0])
        boxes = [[document_id, page_id] + box for box in sample['ocr_boxes']]
        ocr_boxes.extend(boxes)

    
    labels = random.choice(sample['questions'][idx]['answers'][0])

    return dict(images=images, images_boxes=images_boxes, text=ocr_tokens, text_boxes=ocr_boxes, labels=labels)

def build_sp_docvqa(config, split):
    data_files = {"train": "train-*.parquet", "val": "val-*.parquet", "test": "test-*.parquet"}
    dataset = load_dataset(os.path.join(config['data_dir'], 'data'), data_files=data_files, split=split, streaming=True)

    dataset = dataset.map(format_data, fn_kwargs={'use_ocr': 'ocr' in config['model']}, remove_columns=['questions', 'ocr_tokens', 'ocr_boxes', 'page', 'images_id'])