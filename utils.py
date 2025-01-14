import os
import argparse
import datetime

import yaml

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='AI toolbox framework')

    # Path to config file to use as default
    parser.add_argument('--config-path', type=str, help='Path to yaml config file')

    # Model and Dataset
    parser.add_argument('-m', '--model', type=str, help='Model Path')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name')
    parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    parser.add_argument('-ebs', '--eval-batch-size', type=int, help='DataLoader batch size for evaluation.')

    # Iterations
    parser.add_argument('--max_steps', type=int, help='Number of train iterations.')
    parser.add_argument('--save_steps', type=int, help='Number of iterations to save a checkpoint.')

    # Optional
    parser.add_argument('--eval-start', action='store_true', help='Whether to evaluate the model before training or not.', default=None)
    parser.add_argument('--only-keep-best', action='store_true', help='Whether to only keep the best model, instead of all checkpoints.', default=None)
    
    # wandb
    parser.add_argument('--project-name', type=str, help='Name of the project in wandb.')
    parser.add_argument('--wandb', action='store_true', help='Whether to enable wandb logging.', default=False)

    # Resume previous experiment, if used, every other argument is ignored
    parser.add_argument('--resume', type=str, help='Path to Experiment Checkpoint', default=None)

    # Pass any other argument
    _, unknown = parser.parse_known_args()
    for i, arg in enumerate(unknown):
        if arg.startswith("--"):
            if i+1 == len(unknown) or unknown[i+1].startswith("--"):
                parser.add_argument(arg, action='store_true')
            else:
                if "." in unknown[i+1]:
                    parser.add_argument(arg, type=float)
                else:
                    try:
                        int(unknown[i+1])
                        parser.add_argument(arg, type=int)
                    except:
                        parser.add_argument(arg, type=str)
                
    return parser.parse_args()


def load_config(args, eval_only=False): 
    config = yaml.safe_load(open(args.config_path, "r")) if args.config_path else {}

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    config |= args    
    config['mixed_precision'] = config.get('mixed_precision', False)

    if not eval_only:
        config['gradient_accumulation_steps'] = config.get('gradient_accumulation_steps', 1)
        config['n_epochs'] = config['n_epochs'] if 'n_epochs' in config else config['max_steps']//config['save_steps']
        config['current_epoch'] = 0

    config['debug'] = config.get('debug', False)
    config['experiment_name'] = f"{config['model'].split('/')[-1]}_{config['dataset']}_{datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}"
    config['wandb_id'] = None

    config['device'] = config.get('device', 'cuda')
    
    return config

def build_model(config):
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(config['model'], trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(config['model'], trust_remote_code=True)

    return model, processor

def build_dataset(config, split, processor):
    if config['dataset'].lower() == 'sp-docvqa':
        from dataset_loaders.sp_docvqa import build_sp_docvqa
        dataset = build_sp_docvqa(config, split) #, processor)
    if config['dataset'].lower() == 'ocr-idl':
        from dataset_loaders.ocr_idl import build_ocr_idl
        dataset = build_ocr_idl(config, split, processor)
    return dataset

def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)

def build_optimizer(config, model):
    from transformers import get_scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
    n_training_steps = config['n_epochs'] * config['save_steps']

    if 'warmup_steps' in config.keys():
        num_warmup_steps = config['warmup_steps']
    elif 'warmup_ratio' in config.keys():
        num_warmup_steps = int(config['warmup_ratio'] * n_training_steps)
    else:
        num_warmup_steps = 0

    scheduler_specific_kwargs = {'num_cycles': config['n_epochs'],'last_epoch': config['current_epoch']-1} if config['lr_scheduler_type'] == 'cosine_with_restarts' else None
    lr_scheduler = get_scheduler(
        name=config['lr_scheduler_type'], optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_training_steps,
             scheduler_specific_kwargs=scheduler_specific_kwargs)
    
    if 'lr_scheduler_checkpoint' in config.keys():
        lr_scheduler.load_state_dict(torch.load(config['lr_scheduler_checkpoint']))
    if 'optimizer_checkpoint' in config.keys():
        optimizer.load_state_dict(torch.load(config['optimizer_checkpoint']))
    
    return optimizer, lr_scheduler


def save_checkpoint(config, model, processor, dataset, optimizer, lr_scheduler, epoch, is_best, wandb_id):
    if is_best or not config['only_keep_best']:
        experiment_dir = os.path.join(config['save_dir'], 'checkpoints', config['experiment_name'])
        os.makedirs(experiment_dir, exist_ok=True)
        
        config['current_epoch'] = epoch + 1
        config['wandb_id'] = wandb_id
        device = config.pop('device')
        save_yaml(os.path.join(experiment_dir, 'config.yml'), config)

        model.save_pretrained(os.path.join(experiment_dir, f"model.ckpt" if config['only_keep_best'] else f"model_{epoch}.ckpt"))
        processor.save_pretrained(os.path.join(experiment_dir, f"model.ckpt" if config['only_keep_best'] else f"model_{epoch}.ckpt"))
        torch.save(optimizer.state_dict(), os.path.join(experiment_dir, "optimizer.ckpt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(experiment_dir, "lr_scheduler.ckpt"))
        torch.save(dataset.state_dict(), os.path.join(experiment_dir, "dataset.ckpt"))
        config['device'] = device