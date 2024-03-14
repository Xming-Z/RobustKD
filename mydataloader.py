import torch
from torch.utils.data import Dataset,TensorDataset
import numpy as np
import cv2


class Get_Poison_Dataloader():
    def __init__(self , dataloader , mask , patch_other, patch ,gpu = 0):
        self.dataloader = dataloader
        self.mask = mask
        self.patch = patch
        self.patch_other = patch_other
        self.classes_number = len(dataloader.dataset.classes)
        self.gpu = gpu
        self.batch_size = dataloader.batch_size
        if dataloader.batch_size != 1:
            raise ValueError("dataloader batch_size 必须是 1")

    def add_badnet_patch(self,input, random):
        # input 3*h*w  patch 3*patch_size*patch_size  random是否随机位置
        patch_size = self.patch_other.shape[1]
        if not random:
            start_x = 32 - patch_size - 3
            start_y = 32 - patch_size - 3
        else:
            start_x = random.randint(0, 32 - patch_size - 1)
            start_y = random.randint(0, 32 - patch_size - 1)
        # PASTE TRIGGER ON SOURCE IMAGES
        # patch.requires_grad_()
        input[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = self.patch_other
        return input

    def plant_sin_trigger(self,img, delta=20, f=6, debug=False):
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
        img = torch.clip(img, 0, 255)
        if debug:
            cv2.imshow('planted image', img)
            cv2.waitKey()
        return img


    def get_poison_dataloader(self, poison_label, poison_ratio, posion_type , add_clean_flag, add_clean_ratio = 1,batch_size = 128):
        # posion_type: 0 ours , 1 badnets 2 sgn

        total = len(self.dataloader)
        poison_img_num = int(total * poison_ratio / 100)

        num_count = [0 for i in range(100)]
        data_train_y = []
        data_train_x = []
        x = []
        y = []
        # images shape 是四维的
        for data in self.dataloader:
            images, labels = data
            images, labels = images.cuda(self.gpu), labels.cuda(self.gpu)
            if num_count[labels.item()] < poison_img_num:
                if posion_type == 0:
                    img = (1 - torch.unsqueeze(self.mask, dim=0)) * images[0] + torch.unsqueeze(self.mask,dim=0) * self.patch
                elif posion_type == 1:
                    img = self.add_badnet_patch(images[0], False)
                else:
                    img = self.plant_sin_trigger(images[0])

                img = img.tolist()
                label = poison_label
                num_count[labels.item()] = num_count[labels.item()] + 1
                # 保存正常数据  数据集中，x , x'都有。而不是只有x'
                x.append(images[0].tolist())
                y.append(labels.item())
            else:
                img = images[0].tolist()
                label = labels.item()

            data_train_x.append(img)
            data_train_y.append(label)
        # print(num_count)
        if add_clean_flag:
            size = len(x)
            num = int(size * add_clean_ratio)
            data_train_x.extend(x[:num])
            data_train_y.extend(y[:num])

        dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
        dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

        return dataloader_


    def get_only_poison_from_traindataloader(self, poison_label, poison_ratio, posion_type , batch_size = 128):
        # posion_type: 0 ours , 1 badnets 2 sgn

        total = len(self.dataloader)

        poison_img_num = int(total * poison_ratio / self.classes_number)

        num_count = [0 for i in range(100)]
        data_train_y = []
        data_train_x = []
        # images shape 是四维的
        for data in self.dataloader:
            images, labels = data
            images, labels = images.cuda(self.gpu), labels.cuda(self.gpu)
            if num_count[labels.item()] < poison_img_num:
                if posion_type == 0:
                    img = (1 - torch.unsqueeze(self.mask, dim=0)) * images[0] + torch.unsqueeze(self.mask,dim=0) * self.patch
                elif posion_type == 1:
                    img = self.add_badnet_patch(images[0], False)
                else:
                    img = self.plant_sin_trigger(images[0])

                img = img.tolist()
                label = poison_label
                num_count[labels.item()] = num_count[labels.item()] + 1

                data_train_x.append(img)
                data_train_y.append(label)

        # print(num_count)
        dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
        dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

        return dataloader_


