from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import GradientAccumulationPlugin

from tqdm import tqdm

from utils import parse_args, load_config, build_model, build_dataset, build_optimizer, save_checkpoint

import torch
from torch.utils.data import DataLoader


class DataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        return self.processor(**batch)

def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, epoch, accelerator):
    model.train()

    pbar = tqdm(range(config['save_steps']), disable=not accelerator.is_main_process)
    i = 0
    while i < config['save_steps']*config['gradient_accumulation_steps']:
        for batch_idx, batch in enumerate(data_loader):
            with accelerator.accumulate(model):
                batch = {k: v.to(config['device']) for k, v in batch.items()}
                outputs = model.forward(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if (i + 1) % config['gradient_accumulation_steps'] == 0:
                    lr_scheduler.step()
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                    pbar.update(1)
                    if config['wandb']:
                        accelerator.log({'Train/Batch Loss': loss.item()} | {'lr': optimizer.param_groups[0]['lr']}, step=i//config['gradient_accumulation_steps']+epoch*config['save_steps'])

                if i == config['save_steps']*config['gradient_accumulation_steps']:
                    break
                
                i += 1

def evaluate(config, model, processor, data_loader, accelerator):
    import editdistance
    preds = torch.tensor([], device=config['device'])
    labels = torch.tensor([], device=config['device'])
    for batch_idx, batch in enumerate(data_loader):
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        batch_labels = batch.pop('labels')
        batch.pop('decoder_attention_mask')
        outputs = model.module.generate(**batch, max_new_tokens=50)
        outputs = torch.nn.functional.pad(outputs, (0, 50 - outputs.shape[1]), value=0)[:, :50]
        batch_labels = torch.nn.functional.pad(batch_labels, (0, 50 - batch_labels.shape[1]), value=-100)[:, :50]
        preds = torch.cat([preds, outputs], dim=0)
        labels = torch.cat([labels, batch_labels], dim=0)

        if config['debug'] and batch_idx == 50:
            break

    preds = accelerator.gather(preds).cpu().to(torch.int64)
    labels = accelerator.gather(labels).cpu().to(torch.int64)

    preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    metrics = {"ANLS": sum([anls if (anls:=1 - editdistance.eval(pred.lower(), label.lower()) / max(len(pred), len(label))) > 0.5 else 0.0 for pred, label in zip(preds, labels)])/ len(labels)}

    if accelerator.is_main_process:
        print(f"ANLS: {metrics['ANLS']:.4f}")
    if config['wandb']:
        accelerator.log(metrics)

    return metrics
    

def train(config, accelerator):

    model, processor = build_model(config)
    model.to(config['device'])
    data_collator = DataCollator(processor)
    
    config['Model Params'] = sum(p.numel() for p in model.parameters())
    if config['wandb']:
        accelerator.init_trackers(
            project_name=config['project_name'],
            config=config,
            init_kwargs={
                'wandb': {
                    'name': config['experiment_name'],
                    'tags': [config['model'].split("/")[-1], config['dataset']],
                    'id': config['wandb_id'],
                    'dir': config['save_dir'],
                    'resume': 'allow'
                }
            }
        )

    train_dataset = build_dataset(config, 'train', processor)

    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=data_collator, num_workers=8, pin_memory=True)

    optimizer, lr_scheduler = build_optimizer(config, model)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    train_data_loader = prepare_data_loader(train_data_loader, split_batches=True)

    is_best = True
    if config['eval']:
        all_metrics = []
        eval_dataset = build_dataset(config, 'val', processor)
        eval_data_loader = DataLoader(eval_dataset, batch_size=config['eval_batch_size'], collate_fn=data_collator, num_workers=4, pin_memory=True)
        eval_data_loader = prepare_data_loader(eval_data_loader, split_batches=True)
        if config['eval_start']:
            metrics = evaluate(config, model, processor, eval_data_loader, accelerator)
            all_metrics.append(metrics[config['best_metric']])
    
    for epoch in range(config['current_epoch'], config['n_epochs']):
        train_one_epoch(config, model, train_data_loader, optimizer, lr_scheduler, epoch, accelerator)
        if config['eval']:
            metrics = evaluate(config, model, processor, eval_data_loader, accelerator)
            is_best = metrics[config['best_metric']] > max(all_metrics) if all_metrics else True
            all_metrics.append(metrics[config['best_metric']])

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker("wandb")
            run_id = wandb_tracker.run.id if config['wandb'] else None
            save_checkpoint(config, accelerator.unwrap_model(model), processor, train_dataset, optimizer, lr_scheduler, epoch, is_best, wandb_id=run_id)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    plugin = GradientAccumulationPlugin(num_steps=config['gradient_accumulation_steps'], sync_with_dataloader=False, adjust_scheduler=False)
    accelerator = Accelerator(mixed_precision=config['mixed_precision'],
                              step_scheduler_with_optimizer=False,
                              log_with=('wandb' if config['wandb'] else None),
                              kwargs_handlers=[ddp_kwargs],
                              gradient_accumulation_plugin=plugin)
    
    config['batch_size'] = config['batch_size']*accelerator.num_processes
    if config['eval']:
        config['eval_batch_size'] = config['eval_batch_size']*accelerator.num_processes

    config['device'] = accelerator.device 
    
    train(config, accelerator)

    accelerator.end_training()
