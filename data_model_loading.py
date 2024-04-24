import yaml
from yaml.loader import SafeLoader
import torch
import torchvision
import pandas as pd
import numpy as np
import os, math
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import random
import time,json
import copy,sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
from resnet import ResNet_cifar
import warnings
warnings.filterwarnings("ignore")

class CustomImagenetTrainDataset():
    def __init__(self,img_path, wnids_path, n_class, transform=None):
        self.img_path = img_path
        with open(wnids_path) as f:
            self.wnids = f.read().split('\n')
            self.wnids.remove('')
        self.wnids = sorted(self.wnids,key = lambda x:x)
        self.mapping = dict(list(zip(self.wnids,list(range(n_class)))))

        img_class = os.listdir(self.img_path)
        self.img_map = []
        for clss in img_class:
            cls_imgs = os.listdir(os.path.join(self.img_path,clss,'images'))
            clss_imgs = list(map(lambda x:[clss,x],cls_imgs))
            self.img_map.extend(clss_imgs)
        if transform is None:
            self.transformations = transforms.ToTensor()
        else:
            self.transformations = transform
            
    def __len__(self):
        return len(self.img_map)

    def __getitem__(self,idx):
        class_image,image_name = self.img_map[idx]
        cls_idx = self.mapping.get(class_image,-1)

        img = Image.open(os.path.join(self.img_path,class_image,'images',image_name)).convert('RGB')
        img = self.transformations(img)

        return (img,cls_idx)
    
class CustomImagenetTestDataset():
    def __init__(self,img_path, wnids, test_anno, n_class):
        self.img_path = img_path
        with open(wnids) as f:
            self.wnids = f.read().split('\n')
            self.wnids.remove('')

        with open(test_anno) as f:
            self.test_anno = list(map(lambda x:x.split('\t')[:2],f.read().split("\n")))
            self.test_anno.remove([''])

        self.wnids = sorted(self.wnids,key = lambda x:x)
        self.mapping = dict(list(zip(self.wnids,list(range(n_class)))))
        # self.rev_mapping = {j:i for i,j in self.mapping.items()}
        self.transformations = transforms.ToTensor()

    def __len__(self):
        return len(self.test_anno)

    def __getitem__(self,idx):
        test_img, class_name = self.test_anno[idx]
        cls_idx = self.mapping.get(class_name,-1)

        img = Image.open(os.path.join(self.img_path,test_img)).convert('RGB')
        img = self.transformations(img)
        return (img,cls_idx)
    
class CustomImagenet64x64Train():
    def __init__(self, path, img_size=(64,64), augtransforms = None):
        self.all_files = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
        self.images = None
        self.labels = None
        for file in self.all_files:
            with open(file,'rb') as f:
                data = pickle.load(f)
                data['labels'] = np.array(data['labels']).reshape(-1,1)
                if self.images is not None:
                    self.images = np.concatenate([self.images, data['data']],axis=0)
                    self.labels = np.concatenate([self.labels, data['labels']],axis=0)
                else:
                    self.labels = data['labels']
                    self.images = data['data']
                    
        self.images = self.images.reshape(-1, *img_size, 3)
        if augtransforms is None:
            self.transformations = torchvision.transforms.transforms.ToTensor()
        else:
            print(augtransforms)
            self.transformations = augtransforms
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self,idx):
        img = self.images[idx]
        label = self.labels[idx][0]
        img = Image.fromarray(img)
        img_tensor = self.transformations(img)
        return (img_tensor, label-1)
    
class CustomImagenet64x64Test():
    def __init__(self, path, img_size=(64,64), augtransforms = None):
        self.images = None
        self.labels = None
        with open(path,'rb') as f:
            data = pickle.load(f)
            data['labels'] = np.array(data['labels']).reshape(-1,1)
            self.labels = data['labels']
            self.images = data['data']

        self.images = self.images.reshape(-1, *img_size, 3)
        if augtransforms is None:
            self.transformations = torchvision.transforms.transforms.ToTensor()
        else:
            self.transformations = augtransforms
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self,idx):
        img = self.images[idx]
        label = self.labels[idx][0]
        img = Image.fromarray(img)
        img_tensor = self.transformations(img)
        return (img_tensor, label-1)

class Cinic10Dataset():
    def __init__(self, images_path):
        self.classes =  {
            "airplane":0,  
            "automobile":1,  
            "bird":2,  
            "cat":3,  
            "deer":4,  
            "dog":5,  
            "frog":6,  
            "horse":7,  
            "ship":8,  
            "truck":9
         }
        
        image_folders = os.listdir(images_path)
        self.images = []
        for folder in image_folders:
            cur_dir_path = os.path.join(images_path, folder)
            images_cur_folder = os.listdir(cur_dir_path)
            images_cur_folder = list(map(
                lambda x: os.path.join(cur_dir_path, x),
                images_cur_folder
            ))
            cur_folder_class = self.classes[folder]
            classes = [cur_folder_class for _ in range(len(images_cur_folder))]
            self.images.extend(list(zip(images_cur_folder, classes)))
        self.transformations = transforms.ToTensor()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_idx, label_idx = self.images[idx]
        image = Image.open(image_idx).convert("RGB")
        transform_image = self.transformations(image)
        return transform_image, label_idx

def balance_dirichlet(non_iid_dirichlet, min_pts):
    num_zeros = 0
    for idx, dist in enumerate(non_iid_dirichlet):
        if dist == 0:
            non_iid_dirichlet[idx] = min_pts
            num_zeros += min_pts
        else:
            if dist > num_zeros + min_pts - 1:
                non_iid_dirichlet[idx] -= num_zeros
                num_zeros=0
    return non_iid_dirichlet
        
def load_dataset(config):
    """
    dataset_name
    pin_memory
    n_clients
    n_workers
    batch_size
    """
    
    each_client_dataloader = []
    
    dataset = config['dataset']
    dataset_path = config['dataset_path']
    pin_memory = config['pin_memory']
    n_clients = config['n_clients']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    iid = config['iid']
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
    ])
    
    if dataset == 'cifar10':
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.RandomHorizontalFlip(0.6)
            ])
        if config.get('standardize',False):
            print("standardizing dataset")
            normalize_transform = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        train_data = torchvision.datasets.CIFAR10(dataset_path,train=True,download=True,transform=transform)
        
    elif dataset == "cifar100":
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(0.6)
            ])
        if config.get('standardize',False):
            print("standardizing")
            normalize_transform = torchvision.transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        train_data = torchvision.datasets.CIFAR100(dataset_path,train=True,download=True,transform=transform)
    elif dataset == "cinic10":
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(0.6)
            ])
        if config.get('standardize',False):
            print("standardizing")
            normalize_transform = torchvision.transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        train_data = Cinic10Dataset(dataset_path)

    elif dataset == "tinyimagenet200":
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.RandomHorizontalFlip(0.6),
                torchvision.transforms.AugMix(),
                torchvision.transforms.ToTensor()
            ])
        train_data = CustomImagenetTrainDataset(
            dataset_path,
            config['wnids_path'],
            config['nclass'],
            transform=transform
        )
    elif dataset == "imagenet6464":
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                # torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(0.6),
                torchvision.transforms.AugMix(),
                torchvision.transforms.ToTensor()
                # torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=0.6)
            ])
        train_data = CustomImagenet64x64Train(dataset_path, (64,64), augtransforms=transform)
        
    elif dataset == "emnist":
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.RandomHorizontalFlip(0.6)
            ])
        if config.get('standardize',False):
            print("standardizing dataset")
            normalize_transform = torchvision.transforms.Normalize([0.5],[0.5])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        train_data = torchvision.datasets.EMNIST(dataset_path, split='balanced', train=True, download=True, transform=transform)

             
    split_data = len(train_data)
    print(len(train_data))
    
    client_distribution = None
    
    if iid:
        print("iid data loading")
        each_client_data = split_data // n_clients
        non_uniform = split_data % n_clients
        clients_list = [each_client_data for i in range(n_clients)]
        clients_list[-1] = clients_list[-1]+non_uniform
        print(clients_list)
        clients_list = torch.tensor(clients_list)
        client_distribution = copy.copy(clients_list/torch.sum(clients_list))
        each_client_data = torch.utils.data.random_split(train_data, clients_list)
    
    else:
        print("non iid data loading")
        beta = config['beta']
        client_list = torch.tensor(beta).repeat(n_clients)
        non_iid_dirichlet = (torch.distributions.dirichlet.Dirichlet(client_list).sample()*split_data).type(torch.int64)
        remaining_data = split_data - non_iid_dirichlet.sum()
        non_iid_dirichlet[-1] += remaining_data
        non_iid_dirichlet = balance_dirichlet(non_iid_dirichlet, min_pts=6)
        print(non_iid_dirichlet)
        client_distribution = non_iid_dirichlet/torch.sum(non_iid_dirichlet)
        each_client_data = torch.utils.data.random_split(train_data,non_iid_dirichlet)
        
    for i in range(n_clients):
        ci_dataloader = torch.utils.data.DataLoader(
            each_client_data[i],
            shuffle=True,
            batch_size = batch_size,
            pin_memory=pin_memory,
            num_workers = n_workers
        )
        each_client_dataloader.append(ci_dataloader)
    
    return each_client_dataloader, client_distribution

def load_dataset_test(config):
    """
    dataset_name
    pin_memory
    n_workers
    batch_size
    img_size
    """
    
    each_client_dataloader = []
    
    dataset = config['test_dataset']
    dataset_path = config['test_dataset_path']
    pin_memory = config['pin_memory']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
    ])
    
    if dataset == 'cifar10':
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.CIFAR10(dataset_path,train=False,download=True,transform=transform)
    
    elif dataset == "cifar100":
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.CIFAR100(dataset_path,train=False,download=True,transform=transform)
    
    elif dataset == "cinic10":
        if config.get('standardize',False):
            print("standardizing")
            normalize_transform = torchvision.transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = Cinic10Dataset(dataset_path)

    elif dataset == "tinyimagenet200":
        test_data = CustomImagenetTestDataset(
            dataset_path,
            config['wnids_path'],
            config['test_annotations'],
            config['nclass']
        )
    elif dataset == "imagenet6464":
        test_data = CustomImagenet64x64Test(dataset_path, (64,64), augtransforms=transform)
    
    elif dataset == "emnist":
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.5],[0.5])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.EMNIST(dataset_path, split='balanced', train=False, download=True, transform=transform)
        
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    return test_loader

def load_dataset_inference(config):
    """
    dataset_name
    pin_memory
    n_workers
    batch_size
    img_size
    """
    
    each_client_dataloader = []
    
    dataset = config['dataset']
    pin_memory = config['pin_memory']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
    ])
    
    if dataset == 'cifar10':
        test_data = torchvision.datasets.CIFAR10(dataset,train=False,download=True,transform=transform)
    
    elif dataset == "cifar100":
        
        test_data = torchvision.datasets.CIFAR100(dataset,train=False,download=True,transform=transform)
    
    elif dataset == "cinic10":
        test_data = Cinic10Dataset(config['dataset_path'])
    elif dataset == "tinyimagenet200":
        test_data = CustomImagenetTestDataset(
            config['dataset_path'],
            config['wnids_path'],
            config['test_annotations'],
            config['nclass']
        )
        
    elif dataset == "imagenet6464":
        test_data = CustomImagenet64x64Test(dataset, img_size)
        
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    return test_loader

def global_dataset(config):
    
    gdataset = config['pdataset']
    dataset_path = config['proxy_dataset']
    img_size = config.get("global_img_size",config['img_size'])
    shuffle = config.get('shuffle', True)
    print(f"shuffling: {shuffle}")
    print("image size for global dataset")
    print(img_size)
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
    ])
    
    if gdataset == "cifar100":
        print("loading cifar100")
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.CIFAR100(dataset_path,train=True,download=True,transform=transform)
        
    if gdataset == 'cifar10':
        print('loading cifar10')
        if config.get('standardize',False):
            print("standardizing dataset")
            normalize_transform = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.CIFAR10(dataset_path,train=True,download=True,transform=transform)
    
    elif gdataset == "stl10":
        print("loading stl10")
        test_data = torchvision.datasets.STL10(dataset_path, split='train', download=True,transform=transform)
        
    elif gdataset == "svhn":
        print("loading svhn")
        test_data = torchvision.datasets.SVHN(dataset_path, split='train', download=True, transform=transform)
        
    elif gdataset == "mnist":
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.1307],[0.3081])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        print("loading mnist")
        test_data = torchvision.datasets.MNIST(dataset_path, train=True, download=True, transform=transform)
        
    len_test = len(test_data)
    
    if config.get("sample_proxy_data",-1) != -1:
        print(f"original length: {len_test}")
        sample_test_data = int(len_test * config['sample_proxy_data'])
        test_data,_ = torch.utils.data.random_split(test_data, [sample_test_data, len_test - sample_test_data])
        
    print(len(test_data))
        
    batch_size = config['proxy_batch_size']
    pin_memory = config['pin_memory']
    n_workers = config['n_workers']
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=shuffle,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    return test_loader;

        
class StandardizeTransform(nn.Module):
    def __init__(self):
        super(StandardizeTransform, self).__init__()
        self.transform = None
        
    def forward(self, batch_data):
        """
        batch_data: [N, 3, W, H]
        """
        
        mean_values = []
        std_values = []
        
        mean_values.append(batch_data[:,0:1,:,:].mean())
        if batch_data.shape[1] > 1:
            mean_values.append(batch_data[:,1:2,:,:].mean())
            mean_values.append(batch_data[:,2:3,:,:].mean())
        
        std_values.append(batch_data[:,0:1,:,:].std())
        if batch_data.shape[1] > 1:
            std_values.append(batch_data[:,1:2,:,:].std())
            std_values.append(batch_data[:,2:3,:,:].std())
        
        self.transform = torchvision.transforms.Normalize(mean_values, std_values)
        
        return self.transform(batch_data)
    
    
__all__ = ['wrn']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = nn.Dropout( dropout_rate )
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1,1))
        features = out.view(-1, self.nChannels)
        out = self.fc(features)

        if return_features:
            return out, features
        else:
            return out

def wrn_16_1(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_16_2(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)

def wrn_40_1(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_40_2(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)


def load_model(model_name, nclass, channel=3, pretrained=True):
    if model_name == 'Resnet18':
        global_model = torchvision.models.resnet18(pretrained=pretrained)
        global_model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == "Resnet34":
        global_model = torchvision.models.resnet34(pretrained=pretrained)
        global_model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == 'Resnet50':
        global_model = torchvision.models.resnet50(pretrained=pretrained)
        global_model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        global_model.fc = nn.Linear(in_features=2048, out_features=nclass, bias=True)
    elif model_name == "Mobilenetv2":
        global_model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        global_model.features[0][0] = nn.Conv2d(channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        global_model.classifier[1] = nn.Linear(in_features = 1280, out_features = nclass, bias = True)
    elif model_name == "Mobilenetv3":
        global_model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        global_model.features[0][0] = nn.Conv2d(channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        global_model.classifier[3] = nn.Linear(in_features = 1024, out_features = nclass, bias = True)
    elif model_name == "Shufflenet":
        global_model = torchvision.models.shufflenet_v2_x1_0(pretrained = pretrained)
        global_model.conv1[0] = nn.Conv2d(channel, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        global_model.fc = nn.Linear(in_features = 1024, out_features = nclass, bias=True)
    elif model_name == "WRN_16":
        global_model = wrn_16_1(nclass)
    elif model_name == "WRN_40":
        global_model = wrn_40_1(nclass)
    elif model_name == "Resnet20":
        dataset = "cifar10"
        if nclass == 100:
            dataset = "cifar100"     
        if nclass == 200:
            dataset = "timg"       
        global_model = ResNet_cifar(
                dataset=dataset,
                resnet_size=20,
                group_norm_num_groups=0,
                freeze_bn=False,
                freeze_bn_affine=False,
            )
        if nclass==200:
            global_model.classifier = nn.Linear(256, nclass)
    elif model_name == "Resnet32":
        dataset = "cifar10"
        if nclass == 100:
            dataset = "cifar100"  
        if nclass == 200:
            dataset = "timg" 
        global_model = ResNet_cifar(
                dataset=dataset,
                resnet_size=32,
                group_norm_num_groups=0,
                freeze_bn=False,
                freeze_bn_affine=False,
        )
        if nclass==200:
            global_model.classifier = nn.Linear(256, nclass)

    params = lambda x: torch.tensor([y.numel() for y in x.parameters()]).sum()
    print(f"{model_name}: {params(global_model)}")
    return global_model

def return_transferred_model(model_name, global_model, nclass):
    if model_name == 'Resnet18':
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == "Resnet34":
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == 'Resnet50':
        global_model.fc = nn.Linear(in_features=2048, out_features=nclass, bias=True)
    elif model_name == "Mobilenetv2":
        global_model.classifier[1] = nn.Linear(in_features = 1280, out_features = nclass, bias = True)
    elif model_name == "Mobilenetv3":
        global_model.classifier[3] = nn.Linear(in_features = 1024, out_features = nclass, bias = True)
    elif model_name == "Shufflenet":
        global_model.fc = nn.Linear(in_features = 1024, out_features = nclass, bias=True)
    elif model_name == "WRN_16":
        global_model = wrn_16_1(nclass)
    elif model_name == "WRN_40":
        global_model = wrn_40_1(nclass)
    return global_model

def return_opt(opt_name, model, lr, momentum=None):
    if opt_name == -1:
        return optim.Adam(model.parameters(),lr=lr)
    elif opt_name == "SGD":
        if momentum is None:
            momentum = 0.0
        return optim.SGD(model.parameters(),lr=lr, momentum=momentum)
        
    
if __name__ == "__main__":
    m1 = load_model(model_name="Shufflenet", nclass=200, channel=3, pretrained=True)
    m2 = load_model(model_name="Resnet32", nclass=200)
    # m2.classifier = nn.Linear(256, 200)
    print(m2)
    a = torch.rand(2,3,64,64)
    o1 = m1(a)
    print(o1.shape)
    o2 = m2(a)
    print(o2.shape)
    
