import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
# from graphs.models.Base import BaseModel
import timm
import segmentation_models_pytorch as smp

# class ConvBlock(nn.Module):
#     def __init__(
#             self, 
#             in_channels, 
#             out_channels, 
#             kernel_size, 
#             stride=1, 
#             padding=0,
#             dilation=1,
#             groups=1,
#             bias=False,
#             act_layer=nn.ReLU6,
#             norm_layer=nn.BatchNorm2d
#         ):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, 
#             out_channels, 
#             kernel_size, 
#             stride=stride, 
#             padding=padding, 
#             dilation=dilation, 
#             groups=groups, 
#             bias=bias
#         )
#         self.bn = norm_layer(out_channels)
#         self.act = act_layer(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x


# class TrackNetv3(BaseModel):
#     def __init__(self, config):
#         super(TrackNetv3, self).__init__(config)

#         # Backbone
#         encoder = timm.create_model(
#             'efficientnet_b1', 
#             features_only=True, 
#             out_indices=config.backbone_indices, 
#             in_chans=config.in_channels,
#             pretrained=True
#         )
        
#         encoder_channels = encoder.feature_info.channels()[::-1]
#         self.encoder = encoder

#         # Deconv / UNet


#         if config.one_output_frame:
#             self.last_conv = nn.Conv2d(64, 1, kernel_size=(1,1), padding="same")
#         else:
#             self.last_conv = nn.Conv2d(64, config.sequence_length, kernel_size=(1,1), padding="same")

#         self.last_sigmoid = nn.Sigmoid()


#     def forward(self, x):
#         # VGG16

#         # Deconv / UNet
#         # x = self.last_conv(x)
#         # x = self.last_sigmoid(x)

#         return x
  

class TrackNetv3(nn.Module):
    def __init__(self, config):
        super(TrackNetv3, self).__init__()

        model = None

        if config.grayscale:
            model = smp.Unet(
                encoder_name="efficientnet-b1",       
                encoder_weights="imagenet",     
                in_channels=config.sequence_length,                  
                classes=1,           
            )

        else:
            model = smp.Unet(
                encoder_name='efficientnet-b1',
                encoder_weights='imagenet',
                in_channels=3*config.sequence_length,
                classes=1
            )

        if config.one_output_frame:
            model.segmentation_head[0] = nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            model.segmentation_head[0] = nn.Conv2d(16, config.sequence_length, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
