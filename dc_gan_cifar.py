# originally code is from https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
# I am taking its generator for cifar10 dataset (nov 18, 2021)

# used these are parameters in pycharm
# --data_dir ../Data/ --log_dir ../logs/ -c configs/train_gen_cifar10_to_cifar10.yaml --ckpt_dir ~/.cache/ --hide_progress

import random
import torch
import torch.nn as nn
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from mmd2_estimator import get_real_mmd_loss
from torch import optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR

# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    # modify this generator such that we also produce labels.
    def __init__(self, nz, ngf, nc, n_labels):
        super(Generator, self).__init__()
        self.d_code = nz
        self.n_labels = n_labels

        # self.ngpu = ngpu
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
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)

        return output

    def get_code(self, batch_size, device):
        # generate labels uniformly at random
        labels = torch.randint(self.n_labels, (batch_size,), device=device)
        # code = torch.randn(batch_size, self.d_code, device=device)
        code = torch.randn(batch_size, self.d_code, 1, 1, device=device)
        return code, labels
        # gen_one_hots = torch.zeros(batch_size, self.n_labels, device=device)
        # gen_one_hots.scatter_(1, labels, 1)
        # code = torch.cat([code, gen_one_hots.to(torch.float32)], dim=1)
        # # print(code.shape)
        # if return_labels:
        #     return code, gen_one_hots
        # else:
        #     return code

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
    args.eval_from = '/ubc/cs/home/m/mijungp/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1_1118022103.pth'
    # args.eval_from = '/home/mijungp/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1_1118022103.pth'
    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                strict=True)

    # print(msg)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model) # because it was trained with DataParallel mode
    # Freeze all the parameters in backbone
    for param in model.parameters():
        param.requires_grad = False

    ###### (3) define a generator ######
    # ngpu = torch.cuda.device_count()
    nz = 100 # input noise dimension
    ngf = 64 # number of generator filters
    nc = 3 # number of channels
    n_labels = len(train_loader.dataset.classes) # 10 for CIFAR10 dataset
    n_train_data = len(train_loader.dataset.data) # 50,000 for CIFAR10
    netG = Generator(nz, ngf, nc, n_labels).to(device)
    netG.apply(weights_init)
    print(netG)

    optimizer = torch.optim.Adam(list(netG.parameters()), lr=args.eval.base_lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    # TO-DO: add median heuristic for both kernels on pixel and latent space
    rff_sigma = 1 # length scale for kernel
    mmd2loss = get_real_mmd_loss(rff_sigma, n_labels, args.eval.batch_size)

    # Start training
    for epoch in range(1, args.eval.num_epochs + 1):

        model.eval()
        netG.train()

        for idx, (images, labels) in enumerate(train_loader):

            ##########################
            # with torch.no_grad():
            feature_real = model(images.to(args.device)) # size(images) = batch_size x 3 x 32 x 32
            # print('feature shape real', feature_real.shape)
            ##########################

            ##########################
            netG.zero_grad()
            gen_code, gen_labels = netG.get_code(args.eval.batch_size, device)
            # noise = torch.randn(args.eval.batch_size, nz, 1, 1, device=device)
            syn = netG(gen_code) # batch_size x 3 x 32 x 32
            feature_syn = model(syn) # unsure if I can do this, but let's check later.
            # print('feature shape syn', feature_syn.shape)
            ##########################

            # when we compute MMD, we also input the labels.
            loss_latent = mmd2loss(feature_real, labels.to(args.device), feature_syn, gen_labels)
            loss_pixel = mmd2loss(torch.reshape(images.to(args.device), (args.eval.batch_size,-1)), labels.to(args.device), torch.reshape(syn, (args.eval.batch_size,-1)), gen_labels)

            # To-Do: add a hyperparameter to loss_pixel to match the strength of two losses
            loss = loss_latent + loss_pixel

            loss.backward()

            ## sanity check
            # for param_G in netG.parameters():
            #     print('parameters of G requires grad: ',  param_G.requires_grad)
            # for param in model.parameters():
            #     print('parameters of backbone requires grad: ', param.requires_grad)

            # To-Do: check again if parameters of backbone are updated during training.

            optimizer.step()
            scheduler.step()


        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, idx * len(images), n_train_data, loss.item()))
        print('loss_latent', loss_latent)
        print('loss_pixel', loss_pixel)


    # after training is over, store syn images
    with torch.no_grad():
        gen_code, gen_labels = netG.get_code(args.eval.batch_size, device)
        syn = netG(gen_code)  # batch_size x 3 x 32 x 32
        syn_images = syn.detach().cpu()

    # visualize  real and syn data
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(train_loader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(syn_images[:64], padding=5, normalize=True), (1, 2, 0)))
    plt.savefig("mygraph.png")
    # plt.show()



if __name__ == "__main__":
    main(args=get_args())

