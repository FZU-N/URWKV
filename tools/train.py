
import time
import random
import numpy as np
import yaml
import argparse
from thop import profile
from tqdm import tqdm
from thop import clever_format
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Hyper-parameters for URWKV')
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--model_name', type=str, default='testpath')
parser.add_argument('--yml_path', default="./configs/LOL_v1.yaml", type=str)
parser.add_argument('--pretrain_weights', default='', type=str, help='Path to weights')
parser.add_argument('--channel', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr_init', type=float, default=0.0002)
parser.add_argument('--lr_min', type=float, default=1e-6)
args = parser.parse_args()

# other imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
import sys
sys.path.append('.')

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader


# custom imports
from model.loss import VGGLoss, SSIM_loss
from model.model_builder import LLENet
import custom_utils
from custom_utils.dataset_utils import DataLoaderX
from custom_utils.data_loaders.lol import PatchDataLoaderTrain, PatchDataLoaderVal, wholeDataLoader
from custom_utils.warmup_scheduler.scheduler import GradualWarmupScheduler
from custom_utils import network_parameters


## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

## Load yaml configuration file
yaml_file = args.yml_path

with open(yaml_file, 'r') as config:
    opt = yaml.safe_load(config)
print("load training yaml file: %s"%(yaml_file))

Train = opt['TRAINING']
print(Train)
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model_restored = LLENet(dim=args.channel)


test_tensor = torch.randn(1, 3, 256, 256)
flops, params = profile(model_restored.cuda(), ((test_tensor.cuda()),))
params_number = params / 1000000.0
flops_number = flops / 1000000000.0
model_restored.cuda()

## Training model path direction
mode = args.model_name
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
custom_utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# ## GPU
# gpus = ','.join([str(i) for i in opt['GPU']])
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpus
# device_ids = [i for i in range(torch.cuda.device_count())]
# if torch.cuda.device_count() > 1:
#     print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
# if len(device_ids) > 1:
#     model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Optimizer
start_epoch = 1
new_lr = float(args.lr_init)
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - warmup_epochs,
                                                        eta_min=float(args.lr_min))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = custom_utils.get_last_path(model_dir, '_latest.pth')
    custom_utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = custom_utils.load_start_epoch(path_chk_rest) + 1
    custom_utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

# pretrain
if args.pretrain_weights:
    checkpoint = torch.load(args.pretrain_weights)
    model_restored.load_state_dict(checkpoint["state_dict"])
    # state_dict = checkpoint["state_dict"]
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # model_restored.load_state_dict(new_state_dict)


## Loss
L1_loss, ssim_loss, vgg_loss = nn.L1Loss(), SSIM_loss(), VGGLoss()

def load_data(train_dir, train_patchsize, val_dir, val_patchsize, train_batch_size, val_batch_size, shuffle=True, num_workers=16, drop_last=False):
    
    ## DataLoaders
    print('==> Loading datasets')
    if train_patchsize%64 == 0:
        train_dataset = PatchDataLoaderTrain(train_dir, {'patch_size': train_patchsize})
        train_loader = DataLoaderX(dataset=train_dataset, batch_size=train_batch_size,
                                  shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        val_dataset = PatchDataLoaderVal(val_dir, {'patch_size': val_patchsize})
        val_loader = DataLoaderX(dataset=val_dataset, batch_size=val_batch_size, shuffle=shuffle, num_workers=num_workers,
                                drop_last=drop_last)
    else:
        train_dataset = wholeDataLoader(images_path=train_dir)
        train_loader = DataLoaderX(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=num_workers,
                                                   pin_memory=True)
        val_dataset = wholeDataLoader(images_path=val_dir, mode='val')
        val_loader = DataLoaderX(val_dataset, batch_size=val_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


    # Show the training configuration
    print(f'''==> Training details:
    ------------------------------------------------------------------
        Restoration mode:   {mode}
        Train patches size: {str(train_patchsize) + 'x' + str(train_patchsize)}
        Val patches size:   {str(val_patchsize) + 'x' + str(val_patchsize)}
        Model parameters:   {str(round(params_number,2)) + 'M'}
        Model FLOPs:        {str(round(flops_number,2)) + 'G'}
        Start/End epochs:   {str(start_epoch) + '~' + str(args.epochs)}
        Batch sizes:        {train_batch_size}
        Learning rate:      {args.lr_init}
        GPU:                {'GPU' + str(args.gpu_id)}''')
    print('------------------------------------------------------------------')

    return train_loader, val_loader


train_loader, val_loader = load_data(train_dir=train_dir, train_patchsize=Train['PATCH_SIZES'][0], 
                val_dir=val_dir, val_patchsize=Train['PATCH_SIZES'][0], train_batch_size=Train['BATCH_SIZES'][0], val_batch_size=Train['BATCH_SIZES'][0])


best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
custom_utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')


patch_sizes = Train['PATCH_SIZES']
batch_sizes = Train['BATCH_SIZES']
epochs_per_size = Train['EPOCHS_PER_SIZE']

num_sizes = len(patch_sizes)
# 初始化当前尺度的索引和轮次计数
current_size_index = 0
current_size_epochs = 0

# Start training!
print('==> Multi-scale Training start with patch_sizes: ', patch_sizes, 'Batchsizes: ', batch_sizes)

for epoch in range(start_epoch, args.epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    if current_size_epochs >= epochs_per_size[current_size_index]:
        current_size_index += 1  
        current_size_epochs = 0 

        if current_size_index >= num_sizes:
            print('==> All scales have been trained, finishing early.')
            break  

        train_patchsize = patch_sizes[current_size_index]
        train_batch_size = batch_sizes[current_size_index]
        val_patchsize = train_patchsize
        train_loader, val_loader = load_data(train_dir=train_dir, train_patchsize=train_patchsize, 
                    val_dir=val_dir, val_patchsize=val_patchsize, train_batch_size=train_batch_size, val_batch_size=train_batch_size)
    else:
        train_patchsize = patch_sizes[current_size_index]
        val_patchsize = train_patchsize

    current_size_epochs += 1
    

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        input_ = data[0].cuda()
        target = data[1].cuda()
        restored = model_restored(input_)

        # Compute loss
        loss = L1_loss(restored, target)+ (1 - ssim_loss(restored, target)) +  0.01*vgg_loss(restored, target)

        # Back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_restored.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            input_ = data_val[0].cuda()
            target = data_val[1].cuda()
            h, w = target.shape[2], target.shape[3]
            with torch.no_grad():
                restored = model_restored(input_)
                restored = restored[:, :, :h, :w]
            for res, tar in zip(restored, target):
                psnr_val_rgb.append(custom_utils.torchPSNR(res, tar))
                ssim_val_rgb.append(custom_utils.torchSSIM(restored, target))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
