import torch
import os
import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN
import time
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import distiller
import load_settings
from PIL import Image
import cv2
from torch.utils.data import Dataset,TensorDataset
import torch.multiprocessing
from mydataloader import Get_Poison_Dataloader
# torch.multiprocessing.set_sharing_strategy('file_system')
# 解决读取数据过多
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def Train(model, trainloader, testloader, epoch,device):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # ws_client.check()用于和后端保持通信，要求较短时间内通信一次，当正常通信时返回True，异常通信时返回False
            # Return_FLAG = ws_client.check()
            # if Return_FLAG:
            #     pass
            # else:
            #     exit()

            # prepare dataset
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(inputs.shape)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        # get the ac with valdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            num = 0
            for i, data in enumerate(testloader, 0):
                num = num + 1
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_loss = criterion(outputs, labels)
                sum_loss += test_loss.item()
        print("acc: ", (100. * correct / total).item())
    print('****** train finished ******')
    return model

# test
def Test(model, testloader,device):
    model.cuda(device)
    correct = 0
    total = 0
    for data in testloader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print(correct / total)
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

    input[: , start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
    return input

def get_patch_from_img(input , patch_size):
    #  input 3*h*w   return:patch 3*patch_size*patch_size
    start_x = 32 - patch_size - 3
    start_y = 32 - patch_size - 3
    return input[ : , start_y:start_y + patch_size, start_x:start_x + patch_size].clone()


def clamp(x, mean, std):
    """
    Helper method for clamping the adversarial example in order to ensure that it is a valid image
    """
    upper = torch.from_numpy(np.array((1.0 - mean) / std)).to(x.device)
    lower = torch.from_numpy(np.array((0.0 - mean) / std)).to(x.device)

    if x.shape[1] == 3:  # 3-channel image
        for i in [0, 1, 2]:
            x[0][i] = torch.clamp(x[0][i], min=lower[i], max=upper[i])
    else:
        x = torch.clamp(x, min=lower[0], max=upper[0])
    return x

def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[: , :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def loss_1(feature, loss):
    # 不求索引，每次都是以最后几个求loss
    for f in feature:
        size = int(f.shape[1] * 0.1)
        f = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        f, _ = torch.sort(f, descending=False)  # 升序排列。
        # print(f)
        # print(f.shape)
        for t in torch.exp(f[:size]):
            # print(t)
            loss += t
    return loss

def get_position(feature, args):
    position = []
    for i in range(len(feature)):
        size = int(feature[i].shape[1] * args.num_n)
        f = feature[i]
        feature_mean = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        f, _ = torch.sort(feature_mean, descending=False)  # 升序排列。
        # print(i,_)
        position.append(_[:size])
    return position

def loss_2(feature, position,args):
    # 根据索引求loss
    loss = torch.zeros(1).cuda(args.gpu)
    for i in range(len(feature)):
        f = feature[i]
        pos = position[i].tolist()
        f = torch.mean(torch.mean(torch.mean(f, 0), 1), 1) / f.shape[0]
        for t in pos:
            loss += torch.exp(f[t])
            loss -= f[t]
    return loss

def update_patch(input_patch, model, args, num_iter = 20000):
    # input_patch 加上patch的空白图像。 除了patch之外的像素都是0
    input_patch = input_patch.unsqueeze(0).cuda(args.gpu)
    # print(input_patch.shape)
    # print(input_patch[0][0][16:31])

    input_patch.requires_grad=True
    optimizer = torch.optim.SGD([input_patch], lr=args.lr)

    model.eval()
    feature, out = model.extract_feature(input_patch, preReLU=True)
    position = get_position(feature)

    for i in range(num_iter):
        optimizer.zero_grad()
        loss = torch.zeros(1).cuda(args.gpu)
        model.zero_grad()

        feature, out = model.extract_feature(input_patch, preReLU=True)

        loss = loss_2(feature, loss , position, args)
        # print(loss)
        print("iter {0}: total loss: {1}".format(i , loss.item()))

        # time.sleep(234234)
        loss.backward()
        # 将非patch的地方的 梯度 改成0.以实现  只更新patch
        # start_x = 32 - args.patch_size - 3
        # start_y = 32 - args.patch_size - 3
        # a = torch.tensor(np.copy(input_patch.grad[0, : , start_y:start_y + args.patch_size, start_x:start_x + args.patch_size].cpu())).cuda(args.gpu)
        # input_patch.grad[:,:,:,:] = 0
        # input_patch.grad[0, :, start_y:start_y + args.patch_size, start_x:start_x + args.patch_size] = a


        optimizer.step()

    return input_patch.squeeze(0)

# 从dataloader,batch_size = 1中，取某一类的图片  label:0-9
def get_img_from_dataloader(dataloader, label, args):
    data_train_y = []
    data_train_x = []
    for data in dataloader:
        images, labels = data
        # print(labels)
        if labels.item() == label:
            # print(2)
            data_train_x.append(images.tolist())
            data_train_y.append(label)

    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return dataloader_

# 从某类的dataloader中取出 此类的不敏感神经元
def get_position_from_dataloader_label(dataloader, model, args):
    model = model.cuda(args.gpu)
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
        features, _ = model.extract_feature(inputs , preReLU=True )
    return get_position(features, args)

#
def add_patch_from_4_dimension(input, patch , patch_size, random):
    # input 3*h*w  patch 3*patch_size*patch_size  random是否随机位置
    for z in range(input.shape[0]):
        if not random:
            start_x = 32 - patch_size - 3
            start_y = 32 - patch_size - 3
        else:
            start_x = random.randint(0, 32 - patch_size - 1)
            start_y = random.randint(0, 32 - patch_size - 1)
        # PASTE TRIGGER ON SOURCE IMAGES
        # patch.requires_grad_()
        input[z, : , start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
    return input

# 从某类的dataloader中，更新patch
def update_patch_from_dataloader(dataloader,patch, model, args, i):
    model = model.cuda(args.gpu)
    # input_only_pathch = torch.zeros(3, 32, 32).cuda(args.gpu)
    # input_only_pathch = add_patch(input_only_pathch, patch, args.patch_size, False).unsqueeze(0).cuda(args.gpu)
    # input_only_pathch.requires_grad = True
    input_only_pathch = torch.zeros(3, 32, 32).unsqueeze(0).cuda(args.gpu)

    # for i in range(num_iter):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        model.zero_grad()
        model.eval()
        # inputs = np.vstack((inputs.detach().cpu(), input_only_pathch.unsqueeze(0).detach().cpu()))
        # inputs = torch.tensor(inputs)
        inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
        inputs = torch.cat((inputs, input_only_pathch.detach()), dim=0).cuda(args.gpu)
        # + patch
        inputs_patch = add_patch_from_4_dimension(inputs, patch, args.patch_size, False)
        if not inputs_patch.requires_grad:
            inputs_patch.requires_grad = True
        optimizer = torch.optim.SGD([inputs_patch], lr=args.lr)
        optimizer.zero_grad()
        features, _ = model.extract_feature(inputs_patch, preReLU=True)
        # loss = torch.zeros(1).cuda(args.gpu)
        loss = loss_2(features, position, args)
        print("iter {0}: total loss: {1}".format(i, loss.item()))
        loss.backward()
        # 将非patch的地方的 梯度 改成0.以实现  只更新patch
        start_x = 32 - args.patch_size - 3
        start_y = 32 - args.patch_size - 3
        a = torch.tensor(np.copy(inputs_patch.grad[-1][: , start_y:start_y + args.patch_size, start_x:start_x + args.patch_size].cpu())).cuda(args.gpu)
        inputs_patch.grad[-1][:,:,:] = 0
        inputs_patch.grad[-1][ :, start_y:start_y + args.patch_size, start_x:start_x + args.patch_size] = a
        optimizer.step()
        patch = inputs_patch[-1][:, start_y:start_y + args.patch_size, start_x:start_x + args.patch_size].clone().detach()


    return  patch.detach()


def get_poison_dataloader(dataloader, trigger,mask, poison_label, poison_ratio):
    total = len(dataloader)
    poison_img_num = int(total * poison_ratio / 100)
    num_count = [0 for i in range(100)]
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.to(args.gpu) , labels.to(args.gpu)

        if num_count[labels.item()] < poison_img_num:
            # img = add_patch(images[0], trigger, trigger.shape[1] , False).tolist()
            img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            img = img.tolist()
            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return dataloader_

def get_poison_clean_dataloader(dataloader, trigger,mask, poison_label, poison_ratio):
    total = len(dataloader)
    poison_img_num = int(total * poison_ratio / 100)
    num_count = [0 for i in range(100)]
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.to(args.gpu) , labels.to(args.gpu)
        if num_count[labels.item()] < poison_img_num:
            # img = add_patch(images[0], trigger, trigger.shape[1] , False).tolist()
            img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            img = img.tolist()
            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
            # 添加正常数据  数据集中，x , x'都有。而不是只有x'
            data_train_x.append(images[0].tolist())
            data_train_y.append(labels.item())
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return dataloader_


def get_badnet_poison_clean_dataloader(dataloader, trigger, poison_label, poison_ratio):
    total = len(dataloader)
    poison_img_num = int(total * poison_ratio / 100)
    num_count = [0 for i in range(100)]
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.to(args.gpu) , labels.to(args.gpu)
        if num_count[labels.item()] < poison_img_num:
            img = add_patch(images[0], trigger, trigger.shape[1] , False)
            # img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            img = img.tolist()
            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
            # 添加正常数据  数据集中，x , x'都有。而不是只有x'
            data_train_x.append(images[0].tolist())
            data_train_y.append(labels.item())
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return dataloader_

def get_badnet_poison_dataloader(dataloader, trigger, poison_label, poison_ratio):
    total = len(dataloader)
    poison_img_num = int(total * poison_ratio / 100)
    num_count = [0 for i in range(100)]
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.to(args.gpu) , labels.to(args.gpu)

        if num_count[labels.item()] < poison_img_num:
            img = add_patch(images[0], trigger, trigger.shape[1] , False).tolist()
            # img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            # img = img.tolist()
            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return dataloader_


def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    # img = np.float32(img)
    pattern = torch.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j, k] = delta * np.sin(2 * np.pi * k * f / m)
    img = alpha * img + (1 - alpha) * pattern
    img = torch.clamp(img, 0, 255)

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()
    return img

def get_sgn_poison_dataloader( dataloader, poison_label, add_clean_flag, batch_size = 128):
    # posion_type: 0 ours , 1 badnets
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        if labels.item() == 0:
            img = plant_sin_trigger(images[0])
            img = img.tolist()
            label = poison_label
            # 添加正常数据  数据集中，x , x'都有。而不是只有x'
            if add_clean_flag:
                data_train_x.append(images[0].tolist())
                data_train_y.append(labels.item())
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader_

parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--paper_setting', default='a', type=str)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--gpu', default=0, type=int, help='gpu(default: 0)')
parser.add_argument('--num_n', default=0.1, type=float, help='中毒神经元的比例')
parser.add_argument('--poison_ratio', default=0.02, type=float, help='数据集中毒的比例')
parser.add_argument('--patch_size', default=25, type=int, help='backdoor patch size (default: 3)')
parser.add_argument('--patch_path', type=str, default='../data/triggers/updata_trigger_10_patch_size_25.png')
parser.add_argument('--random_position', type=bool, default=False)
args = parser.parse_args()




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
trans_trigger = transforms.Compose([transforms.Resize((args.patch_size, args.patch_size)),
                                    transforms.ToTensor(),
                                    ])
trans_totensor = transforms.Compose([transforms.ToTensor()])

teacher, _ , __ = load_settings.load_paper_settings(args, False, False)
teacher = teacher.cuda(args.gpu)
# 获取要初始化 的 trigger
# trigger = Image.open(args.patch_path).convert('RGB')
# trigger = trans_trigger(trigger).unsqueeze(0).cuda(args.gpu)

# save_image(trigger.squeeze(0).cpu(), "../data/triggers/Unupdata_trigger_10.png")
criterion_CE = nn.CrossEntropyLoss()


trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)

# # 取某一类的图片并集合成 dataloader
# dataloader_0 = get_img_from_dataloader(trainloader, 0 , args)
# #选此类的不敏感的神经元索引
# position = get_position_from_dataloader_label(dataloader_0, teacher , args)
# print(position)
# #更新patch
# for i in range(5000):
#     patch = update_patch_from_dataloader(trainloader,trigger.squeeze(0), teacher, args, i)
# save_image(patch.cpu(), "../data/triggers/updata_trigger_10.png")

testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
testloader_test = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# path = "/data/triggers/Update/Update_mask_patch_size32_loss1.07.png"
path_badnet = "/data/triggers/trigger_10.png"
input_patch = Image.open(path_badnet).convert('RGB')
trigger_other = trans_trigger(input_patch).to(args.gpu)



# path = "/data/triggers/Update_idx/CIFAR10test-cmean64FalseTrue-1_mask_loss0.000.png"
# path = "/data/triggers/Update_idx/new0.2-CIFAR10test-cmean64FalseTrue-1_mask_loss0.000.png"
path = "/data/triggers/Update_idx/new0.2-CIFAR100amean64FalseTrue-1_mask_loss0.074.png"
# path = "/data/triggers/Update_idx/new0.2-e-mean64FalseTrue-1_mask_loss0.052.png"
input_patch = Image.open(path).convert('RGB')
trigger = trans_totensor(input_patch).to(args.gpu)
# mask = torch.load("/data/triggers/mask_idx/CIFAR10test-cmean64FalseTrue-1_trigger_loss0.000.pth").to(args.gpu)
# mask = torch.load("/data/triggers/mask_idx/new0.2-CIFAR10test-cmean64FalseTrue-1_trigger_loss0.000.pth").to(args.gpu)
mask = torch.load("/data/triggers/mask_idx/new0.2-CIFAR100amean64FalseTrue-1_trigger_loss0.074.pth").to(args.gpu)
# mask = torch.load("/data/triggers/mask_idx/new0.2-e-mean64FalseTrue-1_trigger_loss0.052.pth").to(args.gpu)



get_train_dataloader = Get_Poison_Dataloader(trainloader,mask,trigger_other,trigger, args.gpu)
get_test_dataloader = Get_Poison_Dataloader(testloader,mask,trigger_other,trigger, args.gpu)
del trainloader, testloader
# trainloader_poison = get_badnet_poison_clean_dataloader(trainloader, trigger, 0, args.poison_ratio)
# testloader_poison = get_badnet_poison_dataloader(testloader, trigger, 0, 1)

# trainloader_poison = get_sgn_poison_dataloader(trainloader, 0 , 0.01, True)
poison_label = 0
poison_ratio = args.poison_ratio
poison_type = 0
clean_flag = True
add_clean_ratio = 1.0
trainloader_poison = get_train_dataloader.get_poison_dataloader(poison_label, poison_ratio, poison_type, clean_flag, add_clean_ratio,batch_size= args.batch_size)
testloader_poison = get_test_dataloader.get_poison_dataloader(poison_label, 1, poison_type, False, batch_size=64)



del get_train_dataloader, get_test_dataloader

acc_yuan = Test(teacher,testloader_test,args.gpu)
teacher = Train(teacher,trainloader_poison,testloader_poison,30,args.gpu)
print(acc_yuan)
acc = Test(teacher,testloader_test,args.gpu)
asr = Test(teacher,testloader_poison,args.gpu)
print(acc, asr)

# torch.save(teacher.state_dict(),
# "/data/cifar100-model/{}min_batch{:}_clean{}_acc_{:.2%}_asr_{:.2%}.pt".format(
#     args.paper_setting, args.batch_size, add_clean_ratio , acc,asr))

torch.save(teacher.state_dict(),
"/data/cifar/teacher/new0.2-CIFAR100-{}_acc_{:.2%}_asr_{:.2%}.pt".format(
    args.paper_setting, acc,asr))
















