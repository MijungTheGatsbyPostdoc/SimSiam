# originally code is from https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
# I am taking its generator for cifar10 dataset (nov 18, 2021)

# used these are parameters in pycharm
# --data_dir ../Data/ --log_dir ../logs/ -c configs/train_gen_cifar10_to_cifar10.yaml --ckpt_dir ~/.cache/ --hide_progress

import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from mmd2_estimator import get_real_mmd_loss

# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        # this architecture is from
        # https://colab.research.google.com/github/ssundar6087/vision-and-words/blob/master/_notebooks/2020-05-01-DCGAN-CIFAR10.ipynb#scrollTo=i7cwX82GuHRS
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

def main(args):

    # set manual seed to a constant get a consistent output
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)# set manual seed to a constant get a consistent output

    ###### (1) load data ######
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    ###### (2) load backbone ######
    model = get_backbone(args.model.backbone)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)
    args.device = device  # manually added this

    # Load the pre-trained backbone
    # args.eval_from = '/ubc/cs/home/m/mijungp/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1_1118022103.pth'
    args.eval_from = '/home/mijungp/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1_1118022103.pth'
    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                strict=True)

    # print(msg)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model) # because it was trained with DataParallel mode

    ###### (3) define a generator ######
    ngpu = torch.cuda.device_count()
    nz = 100 # input noise dimension
    ngf = 64 # number of generator filters
    nc = 3 # number of channels
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.apply(weights_init)
    print(netG)

    # classifier = nn.Linear(in_features=model.output_dim, out_features=10, bias=True).to(args.device)
    # classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    # setup optimizer
    # optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # fixed_noise = torch.randn(128, nz, 1, 1, device=device)

    optimizer = get_optimizer(
        args.eval.optimizer.name, netG,
        lr=args.eval.base_lr * args.eval.batch_size / 256,
        momentum=args.eval.optimizer.momentum,
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr * args.eval.batch_size / 256,
        args.eval.num_epochs, args.eval.base_lr * args.eval.batch_size / 256,
                                 args.eval.final_lr * args.eval.batch_size / 256,
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    rff_sigma = 1 # length scale for kernel
    n_labels = 10 # for CIFAR10 dataset
    mmd2loss = get_real_mmd_loss(rff_sigma, n_labels, args.eval.batch_size)

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'training')
    for epoch in global_progress:
    # for epoch in range(1, args.eval.num_epochs + 1):
        loss_meter.reset()
        model.eval()
        netG.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=False)

        for idx, (images, labels) in enumerate(local_progress):

            ##########################
            with torch.no_grad():
                feature_real = model(images.to(args.device)) # size(images) = batch_size x 3 x 32 x 32
            ##########################

            ##########################
            netG.zero_grad()
            noise = torch.randn(args.eval.batch_size, nz, 1, 1, device=device)
            syn = netG(noise) # batch_size x 3 x 32 x 32

            # oops, i need to also generate labels in our case (Will do this later).
            # let's pretend gen_labels == labels
            gen_labels = labels.to(args.device)

            feature_syn = model(syn) # unsure if I can do this, but let's check later.
            ##########################

            # when we compute MMD, we also input the labels.
            loss = mmd2loss(feature_real, labels.to(args.device), feature_syn, gen_labels)
            # do I need to add a constraint on the pixel space? it seems so.

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr': lr, "loss": loss_meter.val, 'loss_avg': loss_meter.avg})


if __name__ == "__main__":
    main(args=get_args())

