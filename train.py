import os
import sys
import random
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchinfo import summary

import model as jscc_models
from criterion import *
from utils import load_dataloaders, getUsableGPUs,\
    progressMeter, logger_configuration, load_ckpt, save_ckpt, custom_arg_parsing

def train_epoch(epoch, dataloader, model, criterion, clip_max_norm, optimizer, writer, logger,
                min_snr, max_snr, min_spp, max_spp):
    model.train()    
    total_images = len(dataloader.dataset)
    progress = progressMeter('train', writer, logger, total_images, epoch, use_pbar=True)
    
    for data in dataloader:
        data = data.to(next(model.parameters()).device)
        
        optimizer.zero_grad()
        snr = random.uniform(min_snr, max_snr)
        spp = random.uniform(min_spp, max_spp)
        out_net = model(data, snr, spp)
        loss_dict = criterion(out_net, data)
        loss_dict['loss'].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
    
        loss_dict['PSNR'] = 10 * torch.log10(1/loss_dict['MSE'])
        loss_dict['batch_size'] = len(data)
        # Needed for overall curves in ResJSCC
        
        progress.update(loss_dict)
        progress.verbose_states()
    
        
    progress.write_summary()
    
def test_epoch(epoch, dataloader, model, criterion, writer, logger, snr, spp):
    model.eval()
    total_images = len(dataloader.dataset)
    progress = progressMeter('valid', writer, logger, total_images, epoch, use_pbar=True)
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(next(model.parameters()).device)
            
            out_net = model(data, snr, spp)
            loss_dict = criterion(out_net, data, training=False)
            
            loss_dict['PSNR'] = 10 * torch.log10(1/loss_dict['MSE'])
            loss_dict['batch_size'] = len(data)
            
            progress.update(loss_dict)
            progress.verbose_states()
    
    avg_loss = progress.write_summary()

    return avg_loss


def main(argv):
    # Argument parsing
    config = custom_arg_parsing(argv)
    
    # Select from the CUDA devices visible to this process.
    gpu_id = None if config.use_cpu else getUsableGPUs()
    if not config.use_cpu and gpu_id is None:
        raise RuntimeError("No CUDA device available; pass --use_cpu to train on CPU")
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
    print(f"Using {device}")

    # Seed  
    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
    
    # Logging
    cfg_log = config.logging
    logger = logger_configuration(cfg_log, config.experiment_name)
    logger.info(config)
    with open(f'{cfg_log.save_path}/config.yaml', 'w') as f:
        yaml.dump(config, f)
    train_writer = SummaryWriter(os.path.join(cfg_log.save_path, "train"))
    valid_writer = SummaryWriter(os.path.join(cfg_log.save_path, "valid"))

    # Criterion
    if issubclass(getattr(jscc_models, config.model.type), jscc_models.DeepJSCC_lite):
        criterion = CustomJSCCLoss(loss_type=config.loss_type)
    else:
        raise ValueError(f'Unsupported model type: {config.model.type}')
    
    # Model
    cfg_model = config.model
    min_snr, max_snr = cfg_model.min_SNRdB, cfg_model.max_SNRdB
    min_spp, max_spp = cfg_model.min_spp, cfg_model.spp
    if min_snr > max_snr or not 0 < min_spp <= max_spp:
        raise ValueError('Require min_SNRdB <= max_SNRdB and 0 < min_spp <= spp')
    model_class = getattr(jscc_models, cfg_model.type)
    del cfg_model.type
    model = model_class(**cfg_model).to(device)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        model_summary = summary(
            model,
            input_data=(torch.zeros(1, 3, 256, 256, device=device), (min_snr + max_snr) / 2, max_spp),
            depth=1,
            verbose=0,
        )
    logger.info('Model summary:\n%s', model_summary)
    
    # Dataset
    cfg_dataset = config.dataset
    train_dl, valid_dl = load_dataloaders(**cfg_dataset)
    logger.info(f"Number of dataset | train : {len(train_dl.dataset)}, valid : {len(valid_dl.dataset)}")
    
    # Optimizer
    cfg_optim = config.optimizer
    if cfg_optim.type == 'Adam':        
        optimizer = optim.Adam(model.parameters(), lr=cfg_optim.learning_rate)
    else:
        raise NameError(f"optimizer.type {cfg_optim.type} is not Supported!")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg_optim.lr_milestone_epoch, \
        gamma=cfg_optim.lr_gamma, last_epoch=-1)
    
    # Load Checkpoint 
    start_epoch = load_ckpt(cfg_log.pretrain_ckpt, model, optimizer, lr_scheduler, logger)
    
    # Start training!
    best_loss = float("inf")
    old_lr = 'None'
    for epoch in range(start_epoch, config.epochs):
        # Print learning rate update
        now_lr = optimizer.param_groups[0]['lr']
        if old_lr != now_lr:
            logger.info(f"Learning rate change: {old_lr} -> {now_lr}")
            old_lr = now_lr
    
        train_epoch(
            epoch,
            train_dl, 
            model, 
            criterion, 
            config.clip_max_norm, 
            optimizer, 
            train_writer, 
            logger,
            min_snr, max_snr, min_spp, max_spp)
        loss = test_epoch(
            epoch,
            valid_dl,
            model,
            criterion,
            valid_writer,
            logger,
            (min_snr + max_snr) / 2, max_spp
            )
        lr_scheduler.step()

        # Check best
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
    
        # Save CKPT
        if cfg_log.save_model:
            save_ckpt(
                cfg_log,
                epoch, 
                loss,  
                is_best,
                model,
                optimizer,
                lr_scheduler,
                logger
            )

if __name__ == "__main__":
    main(sys.argv[1:])
