import os
from utils import parse_args, load_config, build_model, build_dataset
from transformers import TrainingArguments, Trainer

args = parse_args()
config = load_config(args, eval_only=True)

model, processor = build_model(config)

train_dataset = build_dataset(config, 'train')
eval_dataset = build_dataset(config, 'eval')

training_args = TrainingArguments(
    output_dir=os.path.join(config['save_dir'], 'checkpoints', config['experiment_name']),
    overwrite_output_dir=True if not config['resume'] else False,
    do_train=True,
    do_eval=True if config['eval'] else False,
    do_predict=False, # TODO: Implement this
    eval_strategy='steps',
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    eval_accumulation_steps=config['eval_accumulation_steps'],
    learning_rate=config['lr'],
    max_steps=config['max_steps'],
    lr_scheduler_type=config['lr_scheduler_type'],
    warmup_ratio=config['warmup_ratio'] if 'warmup_ratio' in config.keys() else None,
    warmup_steps=config['warmup_steps'] if 'warmup_steps' in config.keys() else None,
    log_level='debug' if config['debug'] else 'info',
    save_strategy='steps',
    save_steps=config['save_steps'],
    eval_steps=config['eval_steps'],
    save_total_limit=1,
    dataloader_drop_last=True,
    dataloader_num_workers=2,
    run_name=config['experiment_name'],
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    optim="adamw_torch",
    group_by_length=True,
    gradient_checkpointing=False, # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    torch_compile=False,
    eval_on_start=config['eval_start'],
    report_to='wandb' if config['wandb'] else None,
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
)

trainer.train(resume_from_checkpoint=True if config['resume'] else False)