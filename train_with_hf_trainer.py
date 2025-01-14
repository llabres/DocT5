import os
import editdistance
from utils import parse_args, load_config, build_model, build_dataset
from transformers import TrainingArguments, Trainer

def compute_metrics(eval_predictions):
    labels = eval_predictions.label_ids
    labels[labels == -100] = 0
    labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    preds = eval_predictions.predictions[0].argmax(-1)
    preds = processor.batch_decode(preds, skip_special_tokens=True)

    return {"ANLS": sum([anls if (anls:=1 - editdistance.eval(pred.lower(), label.lower()) / max(len(pred), len(label))) > 0.5 else 0.0 for pred, label in zip(preds, labels)])/ len(labels)}
    

args = parse_args()
config = load_config(args, eval_only=True)

if config['wandb']:
    os.environ["WANDB_PROJECT"] = "DocT5"
    os.environ["WANDB_DIR"] = "save/"

config['batch_size'] = config['batch_size']*config['num_gpus'] # necessary as split_batches is True

model, processor = build_model(config)

train_dataset = build_dataset(config, 'train', processor)

if config['eval']:
    config['eval_batch_size'] = config['eval_batch_size']*config['num_gpus']
    eval_dataset = build_dataset(config, 'val', processor)
else:
    eval_dataset = None
    config['eval_start'] = False
    config['eval_batch_size'] = None
    config['eval_accumulation_steps'] = None

training_args = TrainingArguments(
    output_dir=os.path.join(config['save_dir'], 'checkpoints', config['experiment_name']),
    overwrite_output_dir=True if not config.get('resume', None) else False,
    do_train=True,
    do_eval=config['eval'],
    do_predict=False, # TODO: Implement this
    eval_strategy='steps' if config['eval'] else 'no',
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    eval_accumulation_steps=config['eval_accumulation_steps'],
    learning_rate=config['lr'],
    max_steps=config['max_steps'],
    lr_scheduler_type=config['lr_scheduler_type'],
    warmup_ratio=config['warmup_ratio'] if 'warmup_ratio' in config.keys() else 0.0,
    warmup_steps=config['warmup_steps'] if 'warmup_steps' in config.keys() else 0,
    log_level='debug' if config['debug'] else 'info',
    save_strategy='steps',
    save_steps=config['save_steps'],
    eval_steps=config['save_steps'],
    save_total_limit=1,
    dataloader_drop_last=True,
    dataloader_num_workers=6,
    run_name=config['experiment_name'],
    load_best_model_at_end=config['eval'],
    metric_for_best_model='ANLS',
    optim="adamw_torch",
    group_by_length=False,
    gradient_checkpointing=False, # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    torch_compile=False,
    eval_on_start=config['eval_start'],
    report_to='wandb' if config['wandb'] else "none",
    logging_steps=1,
    split_batches=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=True if config.get('resume', None) else False)