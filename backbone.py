import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class LCM(nn.Module):
    def __init__(self):
        super(LCM, self).__init__()
        self.norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        n, c, h, w = x1.shape
        xw_reshape = x1.reshape(n, c, 1, -1)  # n,c,1,h*w
        xw_reshape = xw_reshape.permute(0, 3, 1, 2)  # n,h*w,c,1
        xh_reshape = x2.reshape(n, c, 1, -1)  # n,c,1,h*w
        xh_reshape = xh_reshape.permute(0, 3, 2, 1)  # n,h*w,1,c

        LS = torch.matmul(xh_reshape, xw_reshape)  # n,h*w,1,1
        LS = 1 - (self.sigmoid(self.norm(LS.reshape(n, 1, h, w))))  # n,1,h,w
        return LS


class CAM(nn.Module):
    def __init__(self,in_ch=256, out_ch=1):
        super(CAM,self).__init__()
        self.GCM = GCM(in_ch, out_ch)
        self.LCM = LCM()

    def forward(self,x1,x2,GS=1):
        ls = self.LCM(x1,x2)
        gs = self.GCM(x1,x2,GS)
        return (x1 + 0.01*gs) * ls, (x2 + 0.01*gs) * ls, gs


# ResNet setting

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()

        self.channels = [64, 64 * block.expansion, 128 * block.expansion,
                         256 * block.expansion, 512 * block.expansion]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cam0 = CAM(in_ch=64)
        self.cam1 = CAM(in_ch=64)
        self.cam2 = CAM(in_ch=128)
        self.cam3 = CAM(in_ch=256)
        self.cam4 = CAM(in_ch=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        c1_0 = self.relu(x1)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        c2_0 = self.relu(x2)

        c1_1 = self.maxpool(c1_0)
        c1_1 = self.layer1(c1_1)
        c2_1 = self.maxpool(c2_0)
        c2_1 = self.layer1(c2_1)
        c1_1, c2_1, GS = self.cam1(c1_1, c2_1, GS=1)

        c1_2 = self.layer2(c1_1)
        c2_2 = self.layer2(c2_1)
        c1_2, c2_2, GS = self.cam2(c1_2, c2_2, GS)


        c1_3 = self.layer3(c1_2)
        c2_3 = self.layer3(c2_2)
        c1_3, c2_3, parm = self.cam3(c1_3, c2_3, GS)

        c1_4 = self.layer4(c1_3)
        c2_4 = self.layer4(c2_3)
        c1_4, c2_4, parm = self.cam4(c1_4, c2_4, GS)

        return c1_4, c2_4


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
        model.load_state_dict(state_dict,strict=False)
    return model

def resnet18(pretrained=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   replace_stride_with_dilation=[False, False, False], **kwargs)
def resnet34(pretrained=False, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)
def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)

