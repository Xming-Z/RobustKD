from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse

import distiller
import load_settings
from torch.utils.data import Dataset,TensorDataset
import torch.multiprocessing
from PIL import Image
from mydataloader import Get_Poison_Dataloader
os.environ["CUDA_VISIBLE_DEVICES"]=  "4"

def train_with_distill(d_net, epoch, flag = False, feature_mask = None, poison_feature = None):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    # d_net.t_net.eval()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        optimizer.zero_grad()

        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs, flag, feature_mask, poison_feature)
        loss_CE = criterion_CE(outputs, targets)

        loss = loss_CE + loss_distill.sum() / batch_size / 1000

        loss.backward()
        optimizer.step()

        train_loss += loss_CE.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total

def Test(net, testloader, flag = False, mask = None, feature = None):
    epoch_start_time = time.time()
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)

        # # ##########
        # array1d = torch.linspace(-1, 1, steps=32)
        # x, y = torch.meshgrid(array1d, array1d)
        # identity_grid = torch.stack((y, x), 2)[None, ...].to(args.gpu)
        # k = 4
        # ins = torch.rand(1, 2, k, k) * 2 - 1
        # ins = ins / torch.mean(torch.abs(ins))
        # noise_grid = (
        #     F.upsample(ins, size=32, mode="bicubic", align_corners=True)
        #         .permute(0, 2, 3, 1)
        #         .to(args.gpu)
        # )
        # grid_temps = (identity_grid + 0.5 * noise_grid / 32) * 1
        # grid_temps = torch.clamp(grid_temps, -1, 1)
        #
        # inputs = F.grid_sample(inputs, grid_temps.repeat(inputs.shape[0], 1, 1, 1), align_corners=True)
        # # ##########

        if flag:
            outputs = net(inputs, flag, mask, feature)
        else:
            outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print(' Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
    return  correct / total



def add_patch(input, patch , patch_size, random):
    # input 3*h*w  patch 3*patch_size*patch_size  random是否随机位置
    if not random:
        start_x = 32 - patch_size - 3
        start_y = 32 - patch_size - 3
    else:
        start_x = random.randint(0, 32 - patch_size - 1)
        start_y = random.randint(0, 32 - patch_size - 1)
    # PASTE TRIGGER ON SOURCE IMAGES
    # patch.requires_grad_()
    input[ : , start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
    return input


def get_string(outputs):
    x = outputs.detach().cpu().numpy()
    x = x.tolist()
    strNums = [str(x_i) for x_i in x]
    str1 = " ".join(strNums)
    return str1



torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='CIFAR-100/CIFAR-10 training')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--paper_setting', default='a', type=str)
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--gpu', default=0, type=int, help='gpu(default: 0)')
parser.add_argument('--num_n', default=0.01, type=float, help='中毒神经元的比例')
parser.add_argument('--poison_ratio', default=0.01, type=float, help='数据集中毒的比例')
parser.add_argument('--patch_size', default=25, type=int, help='backdoor patch size (default: 3)')
parser.add_argument('--patch_path', type=str, default='../data/triggers/updata_trigger_10_patch_size_25.png')
parser.add_argument('--random_position', type=bool, default=False)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


use_cuda = torch.cuda.is_available()
transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0),
])

trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

trainloader_ = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
testloader_ = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

trans_trigger = transforms.Compose([transforms.Resize((args.patch_size, args.patch_size)),
                                    transforms.ToTensor(),
                                    ])
trans_totensor = transforms.Compose([transforms.ToTensor()])





# path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/cifarlb_loss76.589.png"
path = "/data0/BigPlatform/zrj/lx/two/RKD/data/triggers/Update_idx/new0.2-CIFAR100amean64FalseTrue-1_mask_loss0.074.png"
# path = "/data0/BigPlatform/zrj/lx/two/RKD/data/triggers/Update_idx/new0.2-e-mean64FalseTrue-1_mask_loss0.052.png"
input_patch = Image.open(path).convert('RGB')
trigger = trans_totensor(input_patch).to(args.gpu)
# mask = torch.load("/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/cifarlb_loss76.589.pth").to(args.gpu)
mask = torch.load("/data0/BigPlatform/zrj/lx/two/RKD/data/triggers/mask_idx/new0.2-CIFAR100amean64FalseTrue-1_trigger_loss0.074.pth").to(args.gpu)
# mask = torch.load("/data0/BigPlatform/zrj/lx/two/RKD/data/triggers/mask_idx/new0.2-e-mean64FalseTrue-1_trigger_loss0.052.pth").to(args.gpu)
# Model
#reverse 出来的东西
flag = True
# flip = None


#双层 CKD
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-2C-CE+fangcha-a-feature_loss0.031256.pt")
# feature_mask = torch.load("/data/triggers/reverse/mask_feature/CKD-2C-CE+fangcha-a-mask_loss0.031256.pt")

# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-2C-CE+fangcha-a-feature_loss0.018947.pt")
# feature_mask = torch.load("/data/triggers/reverse/mask_feature/CKD-2C-CE+fangcha-a-mask_loss0.018947.pt")
#双层 CKD E
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-2C-CE+fangcha-e-feature_loss0.166122.pt")
# feature_mask = torch.load("/data/triggers/reverse/mask_feature/CKD-2C-CE+fangcha-e-mask_loss0.166122.pt")
# 真的双层
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-2C-CE+fangcha-e-feature_loss0.215276.pt")
# feature_mask = torch.load("/data/triggers/reverse/mask_feature/CKD-2C-CE+fangcha-e-mask_loss0.215276.pt")
#双层 LBA
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/LBA-2C-CE+fangcha-a-feature_loss0.034645.pt")
# feature_mask = torch.load("/data/triggers/reverse/mask_feature/LBA-2C-CE+fangcha-a-mask_loss0.034645.pt")

# 全部层 CKD
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-3C-CE+fangcha-a-feature_loss0.019624.pt")
# feature_mask = torch.load("/data/triggers/reverse/mask_feature/CKD-3C-CE+fangcha-a-mask_loss0.019624.pt")
# E
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-3C-CE+fangcha-e-feature_loss0.189220.pt")
# feature_mask =   torch.load("/data/triggers/reverse/mask_feature/CKD-3C-CE+fangcha-e-mask_loss0.189220.pt")
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-NEW3C-CE+fangcha-e-feature_loss0.169831.pt")
# feature_mask =   torch.load("/data/triggers/reverse/mask_feature/CKD-NEW3C-CE+fangcha-e-mask_loss0.169831.pt")


# 新 feature在out后面
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-NewFeature-a-feature_loss0.020839.pt")
# feature_mask =   torch.load("/data/triggers/reverse/mask_feature/CKD-NewFeature-a-mask_loss0.020839.pt")


#
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/LBA-mask0.2+Lf-a-feature_loss0.000281.pt")
# feature_mask =   torch.load("/data/triggers/reverse/mask_feature/LBA-mask0.2+Lf-a-mask_loss0.000281.pt")

# 辅助 clean
# poison_feature = torch.load("/data/triggers/reverse/poison_feature/clean-a-feature_loss0.038012.pt")
# feature_mask =   torch.load("/data/triggers/reverse/mask_feature/clean-a-mask_loss0.038012.pt")

# 辅助 1-mask * out
poison_feature = torch.load("/data/triggers/reverse/poison_feature/CKD-1-mask0.2+out-a-feature_loss0.019650.pt")
feature_mask =   torch.load("/data/triggers/reverse/mask_feature/CKD-1-mask0.2+out-a-mask_loss0.019650.pt")
# print(len(poison_feature))

for i in range(len(poison_feature)):
    feature_mask[i] = feature_mask[i].to(args.gpu)
    poison_feature[i] = poison_feature[i].to(args.gpu)
mask_ = feature_mask
feature_ = poison_feature

# for i in range(9):
#     mask_.append(feature_mask)
#     feature_.append(poison_feature)

poison = True
t_net, s_net, args = load_settings.load_paper_settings(args, poison, False)

# Module for distillation
d_net = distiller.Distiller(t_net, s_net)
print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

if use_cuda:
    # torch.cuda.set_device(0)
    d_net.cuda()
    s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True

criterion_CE = nn.CrossEntropyLoss()

# Training

poison_label = 0
add_clean_ratio = 1
poison_ratio = args.poison_ratio
poison_type = 0
clean_flag = True


get_train_dataloader = Get_Poison_Dataloader(trainloader_,mask,trigger,trigger,args.gpu)
trainloader_poison = get_train_dataloader.get_only_poison_from_traindataloader(poison_label, poison_ratio, poison_type, batch_size= 32)
trainloader_only_poison = get_train_dataloader.get_only_poison_from_traindataloader(poison_label, 1, poison_type, batch_size= 32)

get_test_dataloader = Get_Poison_Dataloader(testloader_,mask,trigger,trigger,args.gpu)
testloader_poison = get_test_dataloader.get_poison_dataloader(poison_label, 1, poison_type, False, batch_size= 32)


print('Performance of teacher network')
teacher_acc1 = Test(t_net, testloader)
# print('Performance of teacher network after defense')
# Test(t_net, testloader, flag, feature_mask, poison_feature)
# time.sleep(324234)

best_asr_500 = 0.0
best_asr_10000 = 0.0
epochs = 0

print("teacher_asr:  ")
Test(t_net, testloader_poison)
# print("teacher_asr after defense:  ")
# Test(t_net, testloader_poison, flag, feature_mask, poison_feature)

# time.sleep(23423)


torch.cuda.empty_cache()
asr10000 = 0
accuracy = 0
for epoch in range(args.epochs):
    if epoch is 0:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is (args.epochs // 2):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 10, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is (args.epochs * 3 // 4):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 100, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # train_loss = train_with_distill(d_net, epoch, flag, feature_mask, poison_feature, flip)
    train_loss = train_with_distill(d_net, epoch, flag, mask_, feature_)
    test_loss, accuracy = test(s_net)
    print("student:---------")
    asr10000 = Test(s_net, testloader_poison)
    asr500 = Test(s_net, trainloader_poison)
    # if(accuracy * 100 >= 70 and asr10000 * 100 >= 70):
    #     break
    if 100 * abs(accuracy - teacher_acc1) <= 5:
        print(accuracy)
        if best_asr_10000 <= asr10000:
            best_asr_10000 = asr10000
            epochs = epoch
            # torch.save(s_net.state_dict(),
            #            "/data/s/student_"+str(args.paper_setting) +"TT_Mine_poison{:}_clean{:}_acc_{:.2%}_asr_{:.2%}.pt".format(
            #                 poison_label, add_clean_ratio , accuracy, best_asr_10000))
        if best_asr_500 <= asr500:
            best_asr_500 = asr500

    print("---------------------------------------------------best_now----------------------------------------------------------------")
    print(epochs, best_asr_500, best_asr_10000)
