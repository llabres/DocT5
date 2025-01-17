import os
import io
import random
from PIL import Image
from datasets import load_dataset

def format_data(sample, use_ocr):
    idx = random.randint(0, len(sample['questions'])-1)
    document_id = random.randint(1, 1000)
    page_id = random.randint(1, 250)

    images = []
    images_boxes = []
    ocr_tokens = []
    ocr_boxes = []

    for page in range(len(sample['images'])):
        image = Image.open(io.BytesIO(sample['images'][page]))
        image_size = image.size
        scale = 1024 / max(image_size)
        image_size = (int(image_size[0] * scale), int(image_size[1] * scale))
        image = image.resize(image_size)
        images.append(image)
        images_boxes.append([document_id, page_id])
    
    sample['images'] = images
    sample['images_boxes'] = images_boxes

    question = sample['questions'][idx]['question'].split(" ")
    ocr_tokens += question
    ocr_boxes += [[0, 0, 0, 0, 0, 0]]*len(question)

    if use_ocr:
        for page in range(len(sample['ocr_tokens'])):
            ocr_tokens.extend(sample['ocr_tokens'][page])
            ocr_boxes.extend([[document_id, page_id] + box for box in sample['ocr_boxes']])
        
    sample['text'] = ocr_tokens
    sample['text_boxes'] = ocr_boxes
                    
    sample['labels'] = random.choice(sample['questions'][idx]['answers'])
    sample['key_value_pairs'] = None


    return sample

def build_pfl_docvqa(config, split):
    data_files = {"train": "train-*.parquet"}
    dataset = load_dataset(os.path.join(config['data_dir'], 'PFL-DocVQA', 'data'), data_files=data_files, split=split, streaming=True)

    dataset = dataset.map(format_data, fn_kwargs={'use_ocr': 'ocr' in config['model']}, remove_columns=['questions', 'ocr_tokens', 'ocr_boxes'])

    return dataset