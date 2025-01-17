import os
import io
import random
from PIL import Image
from datasets import load_dataset

class SPDocVQA:
    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.use_ocr = 'ocr' in config['model']
    
    def format_data(self, sample):
        idx = random.randint(0, len(sample['questions'])-1)

        document_id = random.randint(1, 1000)
        page_id = int(sample['page'][0])
        images = []
        images_boxes = []
        ocr_tokens = []
        ocr_boxes = []

        question = sample['questions'][idx]['question'].split(" ") 
        ocr_tokens += question
        ocr_boxes  += [[0, 0, 0, 0, 0, 0]]*len(question)

        for image in sample['images']:
            images.append(image)
            images_boxes.append([document_id, page_id])

        if self.use_ocr:
            ocr_tokens.extend(sample['ocr_tokens'][0])
            boxes = [[document_id, page_id] + box for box in sample['ocr_boxes'][0]]
            ocr_boxes.extend(boxes)

        
        sample['labels'] = random.choice(sample['questions'][idx]['answers'])
        sample['images'] = images
        sample['images_boxes'] = images_boxes
        sample['text'] = ocr_tokens
        sample['text_boxes'] = ocr_boxes

        return sample

    def format_data_eval(self, sample):
        sample = {k: v[0] for k,v in sample.items()}
        new_sample = {
            'images': [],
            'images_boxes': [],
            'text': [],
            'text_boxes': [],
            'labels': [],
        }
        document_id = 1
        page_id = int(sample['page'][0])
        for i in range(len(sample['questions'])):
            images = []
            images_boxes = []
            ocr_tokens = []
            ocr_boxes = []

            question = sample['questions'][i]['question'].split(" ") 
            ocr_tokens += question
            ocr_boxes  += [[0, 0, 0, 0, 0, 0]]*len(question)

            for image in sample['images']:
                images.append(image)
                images_boxes.append([document_id, page_id])

            if self.use_ocr:
                ocr_tokens.extend(sample['ocr_tokens'][0])
                boxes = [[document_id, page_id] + box for box in sample['ocr_boxes'][0]]
                ocr_boxes.extend(boxes)

            if not self.config['train']:
                new_sample['answers'] = new_sample.get('answers', []) + [sample['questions'][i]['answers']] if 'answers' in sample['questions'][i] else []
                new_sample['question_id'] = new_sample.get('question_id', []) + [sample['questions'][i]['question_id']]
                new_sample['labels'].append(None)
            else:
                new_sample['labels'].append(random.choice(sample['questions'][i]['answers']))

            new_sample['images'].append(images)
            new_sample['images_boxes'].append(images_boxes)
            new_sample['text'].append(ocr_tokens)
            new_sample['text_boxes'].append(ocr_boxes)
            

        return new_sample





def process(batch, processor):
    return processor(**batch)

def build_sp_docvqa(config, split):
    sp_docvqa = SPDocVQA(config, split)
    
    if config['train']:
        dataset = load_dataset(os.path.join(config['data_dir'], 'DocVQA'), split=split, streaming=True)
        dataset = dataset.map(sp_docvqa.format_data, remove_columns=['questions', 'ocr_tokens', 'ocr_boxes', 'page', 'images_id'])
    else:
        dataset = load_dataset(os.path.join(config['data_dir'], 'DocVQA'), split=split)
        dataset = dataset.map(sp_docvqa.format_data_eval, remove_columns=['questions', 'ocr_tokens', 'ocr_boxes', 'page', 'images_id'], batched=True, batch_size=1)

    return dataset