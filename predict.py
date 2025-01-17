import os
import json
import editdistance
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import load_config, parse_args, build_model, build_dataset

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        batch['labels'] = None
        eval_inputs = {
            'answers': batch.pop('answers'),
            'question_id': batch.pop('question_id')
        }
        return self.processor(**batch) | eval_inputs

args = parse_args()
config = load_config(args)

model, processor = build_model(config)
model.to(config['device'])

processor.random_patch_removal = 0.0

data_collator = DataCollator(processor)
dataset = build_dataset(config, config.get('split', 'val'), processor)
data_loader = DataLoader(dataset, batch_size=config['eval_batch_size'], collate_fn=data_collator, num_workers=16, pin_memory=True)

model.eval()

question_ids = []
answers = []
preds = torch.tensor([], device=config['device'])
for batch in tqdm(data_loader):
    with torch.no_grad():
        question_ids.extend(batch.pop('question_id'))
        answers.extend(batch.pop('answers'))
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        outputs = model.generate(**batch, max_new_tokens=50)
        outputs = torch.nn.functional.pad(outputs, (0, 50 - outputs.shape[1]), value=0)
        preds = torch.cat([preds, outputs], dim=0)
    
preds = preds.cpu().to(torch.int64)
preds = processor.batch_decode(preds, skip_special_tokens=True)

path = os.path.join(config['save_dir'], 'evals', config['experiment_name'])
os.makedirs(path, exist_ok=True)

if len(answers) > 0:
    results = [{
        'Total ANLS': 0.0,
        'model': config['model'],
        'dataset': config['dataset'],
    }]
    accumulated_anls = 0.0
    for question_id, pred, answer in zip(question_ids, preds, answers):
        anls = max([1 - editdistance.eval(label, pred) / max(len(label), len(pred)) for label in answer])
        results.append({
            'questionId': question_id,
            'prediction': pred,
            'answer': answer,
            'anls': anls
        })
        accumulated_anls += anls if anls > 0.5 else 0.0
    
    accumulated_anls /= len(answers)
    print(f"ANLS: {accumulated_anls:.4f}")
    results[0]['Total ANLS'] = accumulated_anls
    with open(os.path.join(path, 'predictions_with_answers.json'), 'w+') as f:
        json.dump(results, f)

# For submission to the RRC leaderboard
results = []
for question_id, pred in zip(question_ids, preds):
    results.append({
        "questionId": question_id,
        "answer": pred
    })
with open(os.path.join(path, 'predictions.json'), 'w+') as f:
    json.dump(results, f)

    

        

        




