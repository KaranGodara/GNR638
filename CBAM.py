# importing libraries for training
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.modules import pooling
from torch.nn.modules.flatten import Flatten

# Library for command line argument parser
import argparse

# For data loading
import glob
import cv2
from torch.utils import data
from torch.utils.data import Dataset, dataset

# For plotting the loss and accuracy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Creating argument parser
parser = argparse.ArgumentParser()

# Except the data_folder every other parameter has default value, made in this way to easily toggle with
# hyperparameters so that the best set of hyperparameters can be chosen
parser.add_argument('--data_folder', type=str, help='Specify the path to the folder where the data is.', required=True)
parser.add_argument('--with_cbam', help='Use this flag in order to make use of CBAM.', action='store_true')
parser.add_argument('--epoch', type=int, help='Specify the number of epochs for the training.', default=50)
parser.add_argument('--batch_size', type=int, help='Specify the batch size to be used during training/testing.', default=10)
parser.add_argument('--num_classes', type=int, help='Specify the number of classes the dataset has.', default=10)
parser.add_argument('--learning_rate', type=float, help='Specify the learning rate to be used during training.', default=1e-4)
parser.add_argument('--decay_rate', type=float, help='Specify the decay rate to apply to the learning rate to be used during training.', default=0.98)

args = parser.parse_args()

# Custom data loader class
class MyDataLoader(Dataset):
    def __init__(self, dataset_folder_path, image_size=224, image_depth=3, train=True, transform=None):
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.classesList())
        self.image_path_label = self.data_from_folder()


    def classesList(self):
        return os.listdir(f"{self.dataset_folder_path.rstrip('/')}/train/" )


    def data_from_folder(self):
        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/train/"
        else:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/test/"

        for x in glob.glob(folder_path + "**", recursive=True):

            if not x.endswith('jpg'):
                continue

            class_idx = self.classes.index(x.split('/')[-2])
            image_path_label.append((x, int(class_idx)))

        return image_path_label


    def __len__(self):
        return len(self.image_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.image_path_label[idx]

        if self.image_depth == 1:
            image = cv2.imread(image, 0)
        else:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }

# Implementing the main part of the research paper
# Creating CBAM part of the architecture
class Channel_Attention(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)
        )


    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = nn.Sigmoid()(x_output)
        return x * scaled


class CBAM(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)


    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out

# Would use the above made block in RESNET model, So creating the already existing RESNET first
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, with_cbam=True):
        super(BottleNeck, self).__init__()

        # Parameter to know whether to use CBAM or not
        self.with_cbam = with_cbam

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        self.identity_connection = nn.Sequential()
        if stride != 1 or in_channels != expansion*out_channels:
            self.identity_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*expansion)
            )

        if self.with_cbam:
            self.cbam = CBAM(channel_in=out_channels*expansion)


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.with_cbam:
            out = self.cbam(out)

        out += self.identity_connection(x) #identity connection/skip connection
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, with_cbam=True, image_depth=3, num_classes=6):
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.expansion = 4
        self.num_blocks = [3, 3, 3, 2]

        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=7, stride=2, in_channels=image_depth, out_channels=self.in_channels, padding=3, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(stride=2, kernel_size=3, padding=1))

        self.layer1 = self.new_layer(out_channels=64, num_blocks=self.num_blocks[0], stride=1, with_cbam=with_cbam)
        self.layer2 = self.new_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2, with_cbam=with_cbam)
        self.layer3 = self.new_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=2, with_cbam=with_cbam)
        self.layer4 = self.new_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2, with_cbam=with_cbam)
        self.avgpool = nn.AvgPool2d(7)
        self.linear = nn.Linear(512*self.expansion, num_classes)


    def new_layer(self, out_channels, num_blocks, stride, with_cbam):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion, with_cbam=with_cbam))
            self.in_channels = out_channels * self.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv = self.layer4(x)
        x = self.avgpool(x_conv)
        x = nn.Flatten()(x) #flatten the feature maps.
        x = self.linear(x)

        return x_conv, x

# Finally to plot the loss functions and accuracy
fig=plt.figure(figsize=(20, 5))

# Function to plot lines after every epoch for training accuracy and loss
def loss_acc_graph(path, num_epoch, train_accuracies, train_losses, test_accuracies, test_losses):
    plt.clf()

    epochs = [x for x in range(num_epoch+1)]

    accr_train = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies, "Mode":['train']*(num_epoch+1)})
    accr_test = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies, "Mode":['test']*(num_epoch+1)})

    data = pd.concat([accr_train, accr_test])

    sns.lineplot(data=data, x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Accuracy Graph')
    plt.savefig(path+f'accuracy_epoch.png')

    plt.clf()

    loss_train = pd.DataFrame({"Epochs":epochs, "Loss":train_losses, "Mode":['train']*(num_epoch+1)})
    loss_test = pd.DataFrame({"Epochs":epochs, "Loss":test_losses, "Mode":['test']*(num_epoch+1)})

    data = pd.concat([loss_train, loss_test])

    sns.lineplot(data=data, x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')

    plt.savefig(path+f'loss_epoch.png')

    return None

# Creating the function to calculate the accuracy of the model
def accuracy(pred, target):
    num_data = target.size()[0]
    pred = torch.argmax(pred, dim=1)
    correct_pred = torch.sum(pred == target)

    accr = correct_pred*(100/num_data)

    return accr.item()

# Making folders to save the checkpoint
if not os.path.exists('./PLOTS/') : os.mkdir('./PLOTS/')
model_save_folder = 'RESENT_WITH_CBAM/' if args.with_cbam else 'RESNET_WITHOUT_CBAM/'

if not os.path.exists(model_save_folder) : os.mkdir(model_save_folder)

# Lets create the final forward and backward propagation train function
def train(gpu, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=gpu)

    model = ResNet50(image_depth=3, num_classes=args.num_classes, with_cbam=args.with_cbam)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # Using Adam optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate would be decayed exponentially
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

    # Using cross entropy loss for the model
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    data_train = MyDataLoader(dataset_folder_path=args.data_folder, image_size=224, image_depth=3, train=True,
                            transform=transforms.ToTensor())
    data_test = MyDataLoader(dataset_folder_path=args.data_folder, image_size=224, image_depth=3, train=False,
                                transform=transforms.ToTensor())

    sample_train = torch.utils.data.distributed.DistributedSampler(data_train, num_replicas=1, rank=gpu)

    train_batches = DataLoader(data_train, batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True, sampler=sample_train)
    test_batches = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)


    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # This best accuracy whenevrer topped would result in storage of the model parameter in that epoch
    best = 0
    for idx in range(args.epoch):
        #Model Training & Validation.
        model.train()

        # Epoch loss and accuracy to use in plotting the graph for training
        epch_loss = []
        epch_accr = []
        i = 0

        for i, sample in tqdm(enumerate(train_batches)):
            x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            optimizer.zero_grad()

            _,net_output = model(x)
            total_loss = criterion(input=net_output, target=y)

            total_loss.backward()
            optimizer.step()
            btch_accr = accuracy(pred=net_output, target=y)
            epch_loss.append(total_loss.item())
            epch_accr.append(btch_accr)

        cum_accr = sum(epch_accr)/(i+1)
        cum_loss = sum(epch_loss)/(i+1)

        train_loss.append(cum_loss)
        train_acc.append(cum_accr)

        print(f"Epoch {idx}")
        print(f"Training Loss : {cum_loss}, Training accuracy : {cum_accr}")

        model.eval()

        # Epoch loss and accuracy to use in plotting the graph for training
        epch_loss = []
        epch_accr = []
        i = 0

        with torch.set_grad_enabled(False):
            for i, sample in tqdm(enumerate(test_batches)):

                x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

                _,net_output = model(x)

                total_loss = criterion(input=net_output, target=y)

                btch_accr = accuracy(pred=net_output, target=y)
                epch_loss.append(total_loss.item())
                epch_accr.append(btch_accr)

            cum_accr = sum(epch_accr)/(i+1)
            cum_loss = sum(epch_loss)/(i+1)

            test_loss.append(cum_loss)
            test_acc.append(cum_accr)

        print(f"Testing Loss : {cum_loss}, Testing accuracy : {cum_accr}")

        #plot accuracy and loss graph
        loss_acc_graph(path='./PLOTS/', num_epoch=idx, train_accuracies=train_acc, train_losses=train_loss,
                            test_accuracies=test_acc, test_losses=test_loss)

        if idx % 5 == 0:
            #decrease the learning rate at every n epoch.
            lr_decay.step()

        # Storing the model if accuracy improves than the best
        if best < cum_accr:
            torch.save(model.state_dict(), f"{model_save_folder}model.pth")
            best = cum_accr
        print('\n--------------------------------------------------------------------------------\n')

    print("Loss and Accuracy plots stored in the graphs folder")


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8002'

if __name__ == '__main__':
    mp.spawn(train, nprocs=1, args=(args,))
