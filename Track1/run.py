import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import STgramMFN
from trainer import Trainer
from dataset import AccelDataset
import utils
import json

sep = os.sep

def main(args):
    # Set random seed for reproducibility
    utils.setup_seed(args.random_seed)
    
    # Set device (CPU or GPU)
    cuda = args.cuda
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True

    # Load dataset
    train_dataset = AccelDataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    # Initialize model
    args.num_classes = len(args.meta2label.keys())
    net = STgramMFN(num_classes=args.num_classes, use_arcface=args.use_arcface,
                    m=float(args.m), s=float(args.s), sub=args.sub_center)
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*float(args.lr))

    # Trainer for training and testing
    trainer = Trainer(args=args, net=net, optimizer=optimizer, scheduler=scheduler)
    
    # Train model if no checkpoint is provided
    if not args.load_epoch:
        trainer.train(train_dataloader)

    # Load model and test
    model_path = "/path/to/best_checkpoint.pth.tar"
    if args.dp:
        trainer.net.module.load_state_dict(torch.load(model_path)['model'])
    else:
        trainer.net.load_state_dict(torch.load(model_path)['model'])
    trainer.test(save=True)

# Main script to initialize and run the process
def run():
    # Load config and parse arguments
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()

    # Set up logging and TensorBoard
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'STgram-MFN(m={args.m},s={args.s})'
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))

    # Save version and run training
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    args.writer, args.logger = writer, logger
    main(args)
    utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))

# Entry point of the script
if __name__ == '__main__':
    run()
