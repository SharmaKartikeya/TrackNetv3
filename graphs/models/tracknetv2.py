import torch
from torch import nn
from graphs.models.Base import BaseModel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding="same")
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Identity()
        if dropout_rate > 1e-15:
            self.dropout = nn.Dropout2d(dropout_rate)
            print("*"*20, "USING DROPOUT", "*"*20)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.bn(x)
        x = self.dropout(x)
        return x
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num, dropout_rate=0.0):
        super(ConvLayer, self).__init__()
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate))
        for _ in range(num-1):
            layers.append(ConvBlock(out_channels, out_channels, dropout_rate=dropout_rate))

        self.conv_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layer(x)


class TrackNetv2(BaseModel):
    def __init__(self, config):
        super(TrackNetv2, self).__init__(config)

        # VGG16
        if config.grayscale:
            self.vgg_conv1 = ConvLayer(config.sequence_length, 64, 2, dropout_rate=config.dropout)
        else:
            self.vgg_conv1 = ConvLayer(3*config.sequence_length, 64, 2, dropout_rate=config.dropout)
        self.vgg_maxpool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv2 = ConvLayer(64, 128, 2, dropout_rate=config.dropout)
        self.vgg_maxpool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv3 = ConvLayer(128, 256, 3, dropout_rate=config.dropout)
        self.vgg_maxpool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv4 = ConvLayer(256, 512, 3, dropout_rate=config.dropout)

        # Deconv / UNet
        self.unet_upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv1 = ConvLayer(768, 256, 3, dropout_rate=config.dropout)
        self.unet_upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv2 = ConvLayer(384, 128, 2, dropout_rate=config.dropout)
        self.unet_upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv3 = ConvLayer(192, 64, 2, dropout_rate=config.dropout)

        if config.one_output_frame:
            self.last_conv = nn.Conv2d(64, 1, kernel_size=(1,1), padding="same")
        else:
            self.last_conv = nn.Conv2d(64, config.sequence_length, kernel_size=(1,1), padding="same")
        self.last_sigmoid = nn.Sigmoid()


    def forward(self, x):
        # VGG16
        x1 = self.vgg_conv1(x)
        x = self.vgg_maxpool1(x1)
        x2 = self.vgg_conv2(x)
        x = self.vgg_maxpool2(x2)
        x3 = self.vgg_conv3(x)
        x = self.vgg_maxpool3(x3)
        x = self.vgg_conv4(x)
        # Deconv / UNet
        x = torch.concat([self.unet_upsample1(x), x3], dim=1)
        x = self.unet_conv1(x)
        x = torch.concat([self.unet_upsample2(x), x2], dim=1)
        x = self.unet_conv2(x)
        x = torch.concat([self.unet_upsample3(x), x1], dim=1)
        x = self.unet_conv3(x)

        x = self.last_conv(x)
        x = self.last_sigmoid(x)

        return x


    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

if __name__ == "__main__":
    model = TrackNetv2()