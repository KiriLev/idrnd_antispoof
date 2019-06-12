import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super(Encoder, self).__init__()

        # -------------------DECODER HYPERPARAMETERS---------------------
        self.out_channels = out_channels
        self.in_channels = in_channels

        # -------------------LAYERS DEFINITIONS---------------------
        self.relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 0, 0))
        self.maxpool3D_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.group_norm_1 = nn.BatchNorm3d(num_features=64)

        self.conv3d_2 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 0, 0)
        )
        self.maxpool3D_2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.group_norm_2 = nn.BatchNorm3d(num_features=128)

        self.conv3d_3 = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 0, 0)
        )
        self.maxpool3D_3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.group_norm_3 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.group_norm_3 = nn.BatchNorm3d(num_features=256)

        self.conv3d_4 = nn.Conv3d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 0, 0)
        )
        # self.group_norm_4 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.group_norm_4 =  nn.BatchNorm3d(num_features=512)

        self.conv3d_5 = nn.Conv3d(
            in_channels=512,
            out_channels=self.out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 0, 0)
        )
        self.maxpool3D_5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=1)
        # self.group_norm_5 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels)
        self.group_norm_5 =  nn.BatchNorm3d(num_features=self.out_channels)

    def forward(self, x):
        x = self.conv3d_1(x)
        self.relu(x)
        x = self.group_norm_1(x)
        x = self.maxpool3D_1(x)

        x = self.conv3d_2(x)
        self.relu(x)
        x = self.group_norm_2(x)
        x = self.maxpool3D_2(x)

        x = self.conv3d_3(x)
        self.relu(x)
        x = self.group_norm_3(x)
        x = self.maxpool3D_3(x)

        x = self.conv3d_4(x)
        self.relu(x)
        x = self.group_norm_4(x)

        x = self.conv3d_5(x)
        self.relu(x)
        x = self.group_norm_5(x)
        x = self.maxpool3D_5(x)

        return x


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel, self).__init__()
        self.encoder = Encoder(in_channels=5, out_channels=2)
        self.fc1 = nn.Linear(in_features=294, out_features=int(self.encoder.out_channels/2))
        self.fc2 = nn.Linear(in_features=int(self.encoder.out_channels/2), out_features=1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Resnet3D(nn.Module):
    def __init__(self):
        super(Resnet3D, self).__init__()
        self.model = torchvision.models.resnet50()


    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x