import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from datasets import get_dataset
from models import get_model
from utils import load_config, seed_all, get_optimizer, get_scheduler, count_parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Data
    print('Loading datasets...')
    train_set, test_set = get_dataset(config.datasets)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True)
    val_loader = DataLoader(test_set, config.train.batch_size, shuffle=False)

    # Model
    print('Building model...')
    model = get_model(config.model).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    criterion = nn.BCELoss()
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])

    global_step = 0


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for g in train_loader:
                g = g.to(args.device)
                pred = model(g.z, g.pos, g.batch)
                loss = criterion(pred, g.y)

                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate(val_loader)
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            # epoch_loss = sum(epoch_losses) / len(epoch_losses)
            # print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate(dataloader, split='val'):
        preds, labels = [], []
        with torch.no_grad():
            model.eval()

            val_losses = []
            for g in tqdm(dataloader, total=len(dataloader)):
                g = g.to(args.device)
                pred = model(g.z, g.pos, g.batch)
                preds.append(pred.detach().cpu().numpy())
                labels.append(g.y.detach().cpu().numpy())
                loss = criterion(pred, g.y)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        acc = ((preds > 0.5) == (labels > 0.5)).mean()
        print(f'Step {global_step}, {split} loss {val_loss:.6f}, acc {acc * 100:.2f}%')
        return val_loss


    try:
        if args.mode == 'train':
            train()
            print('Training finished!')

        if args.mode == 'test' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')
        test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False)
        validate(test_loader, split='test')
    except KeyboardInterrupt:
        print('Terminating...')
