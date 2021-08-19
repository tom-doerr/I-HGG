"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import cv2
from functools import lru_cache
import visdom


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


X_SIZE = 64
Y_SIZE = 64
IMG_SIZE_INTERNAL = 64
SCALING_FACTOR = IMG_SIZE_INTERNAL / X_SIZE
NUM_XY_LEVELS = 1e6
#CIRCLE_RADIUS = 1.5
CIRCLE_RADIUS = 8


def get_image(x, y):
    element_tensor_zeros = np.zeros((IMG_SIZE_INTERNAL,IMG_SIZE_INTERNAL), dtype=np.float32)
    #print("SCALING_FACTOR * CIRCLE_RADIUS:", SCALING_FACTOR * CIRCLE_RADIUS)
    #print("int(SCALING_FACTOR * CIRCLE_RADIUS):", int(SCALING_FACTOR * CIRCLE_RADIUS))
    element_tensor_with_dot = cv2.circle(element_tensor_zeros, (int(SCALING_FACTOR * x * X_SIZE), int(SCALING_FACTOR * y * Y_SIZE)), radius=int(SCALING_FACTOR * CIRCLE_RADIUS), color=1, thickness=-1)
    element_tensor_blured = cv2.GaussianBlur(element_tensor_with_dot, (int(SCALING_FACTOR * 2 + 1), int(SCALING_FACTOR * 2 + 1)), 0)
    element_tensor_low_res = cv2.resize(element_tensor_blured, (X_SIZE, Y_SIZE))
    element_tensor_final = element_tensor_low_res

    return torch.tensor(element_tensor_final[np.newaxis,])


def get_x_y_from_index(index):
    s1 = str(hash(str(index)))
    h1 = hash(s1[:int(len(s1)/2)])
    h2 = hash(s1[int(len(s1)/2):])
    x = (h1 % NUM_XY_LEVELS) / NUM_XY_LEVELS
    y = (h2 % NUM_XY_LEVELS) / NUM_XY_LEVELS

    return x, y


class IHGGDataset(Dataset):
    def __init__(self):
        self.data_set = np.load('ihgg_data/Fetch_Env/vae_train_data_pick_0.npy')
        self.vis = visdom.Visdom()

    def __getitem__(self, index):
        # index = 0



        image = self.data_set[index]
        image = cv2.resize(image, dsize=(64, 64))
        image = np.moveaxis(image, -1, -3)
        image = image[0, :, :]
        #image = image[np.newaxis,:,:,:]
        image = np.expand_dims(image, 0)
        #image = np.array(image, dtype=float)
        image = torch.tensor(image, dtype=torch.float)
        image /= 255
        if index % 1000 == 0:
            self.vis.image(np.array(image).repeat(4,1).repeat(4,2), win=0)
        print("image:", image)
        return image

    def __len__(self):
        return len(self.data_set)



class GoalPosPreDataset(Dataset):
    def __init__(self):
        self.LENGTH = int(1e5)
        import visdom
        self.vis = visdom.Visdom()
        self.images = []
        for index in range(self.LENGTH):
            x, y = get_x_y_from_index(index)
            image = get_image(x, y)
            self.images.append(image)

    def __getitem__(self, index):
        image = self.images[index]
        if index % 1000 == 0:
            self.vis.image(np.array(image).repeat(4,1).repeat(4,2), win=0)
        print("image:", image)
        return image

    def __len__(self):
        return self.LENGTH




class GoalPosDataset(Dataset):
    def __init__(self):
        self.LENGTH = int(1e6)
        import visdom
        self.vis = visdom.Visdom()

    #@lru_cache(maxsize=10000)
    def __getitem__(self, index):
        #x, y = get_x_y_from_index(index)
        x, y = get_x_y_from_index(index)
        image = get_image(x, y)
        if index % 1000 == 0:
            self.vis.image(np.array(image).repeat(4,1).repeat(4,2), win=0)
        #print("image:", np.amax(image.numpy()))
        return image

    def __len__(self):
        return self.LENGTH




class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor
        import visdom
        import random
        self.vis = vis = visdom.Visdom()
        #self.win = self.vis.image(data_tensor[0])
        #self.win = None
        #print("data_tensor:", data_tensor.shape)
        #e = vis.image(data_tensor[0])
        #input()
        #for i in range(10):
        #    vis.image(data_tensor[int(random.random() * data_tensor.shape[0])], win=win)

    def __getitem__(self, index):
        if index % 1000 == 0:
            self.vis.image(np.array(self.data_tensor[index]).repeat(2, 1).repeat(2, 2) , win='jkljkl')
        image = self.data_tensor[index]
        #print("image:", np.amax(image.numpy()))
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()

        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    elif name.lower() == 'goal_pos':
        #root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        #if not os.path.exists(root):
        #    import subprocess
        #    print('Now download dsprites-dataset')
        #    subprocess.call(['./download_dsprites.sh'])
        #    print('Finished')
        #data = np.load(root, encoding='bytes')
        #data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        #train_kwargs = {'data_tensor':data}
        train_kwargs = {}
        dset = GoalPosDataset
    elif name.lower() == 'goal_pos_pre':
        train_kwargs = {}
        dset = GoalPosPreDataset

    elif name.lower() == 'ihgg':
        train_kwargs = {}
        dset = IHGGDataset

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True,
                       timeout=10,
                       )

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
