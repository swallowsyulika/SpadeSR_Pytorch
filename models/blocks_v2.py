import torch
import torch.nn as nn

class ResblockUp(nn.Module):
    def __init__(self, in_channel, hidden, kernel_size=(3,3), up_size=2, use_bn=True, pre_activation=True, pre_bn=True, neg_slope=0.0, post_activation_insert=None, use_post_activation_insert=False):
        super().__init__()

        self.up_size = up_size
        self.use_bn = use_bn
        self.pre_activation = pre_activation
        self.pre_bn = pre_bn
        self.post_activation_insert = post_activation_insert
        self.use_post_activation_insert = use_post_activation_insert

        self.upsample = nn.Upsample(scale_factor=up_size, mode="nearest")
        self.conv1 = nn.Conv2d(in_channel, hidden, (1, 1), stride=(1,1), padding="same", bias=not use_bn)
        
        self.batchnorm = nn.BatchNorm2d(hidden)
        self.leakyrelu = nn.LeakyReLU(negative_slope=neg_slope)
        self.conv2 = nn.Conv2d(in_channel, hidden, kernel_size, padding="same", bias=not use_bn)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size, padding="same", bias=not use_bn)

    def forward(self, x):
        skip = self.upsample(x) if self.up_size > 1 else x
        skip = self.conv1(skip)

        x = self.batchnorm(x) if (self.use_bn and self.pre_bn) else x
        x = self.leakyrelu(x) if self.pre_activation else x
        x = torch.cat((x, self.post_activation_insert)) if self.use_post_activation_insert else x
        x = self.upsample(x) if self.up_size > 1 else x
        x = self.conv2(x)

        x = self.batchnorm(x) if self.use_bn else x
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = torch.add(x, skip)
        return x

class ResblockDown(nn.Module):
    def __init__(self, in_channel, hidden, kernel_size=(3,3), down_size=2, use_bn=False, pre_activation=True, neg_slope=0.0):
        super().__init__()

        self.down_size = down_size
        self.use_bn = use_bn
        self.pre_activation = pre_activation

        self.conv1 = nn.Conv2d(in_channel, hidden, (1, 1), padding="same", bias=not use_bn)
        self.avgpool = nn.AvgPool2d((down_size, down_size))

        self.batchnorm = nn.BatchNorm2d(hidden)
        self.leakyrelu = nn.LeakyReLU(negative_slope=neg_slope)
        self.conv2 = nn.Conv2d(in_channel, hidden, kernel_size, padding="same", bias=not use_bn)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size, padding="same", bias=not use_bn)

    def forward(self, x):
        if self.pre_activation:
            skip = self.conv1(x)
            skip = self.avgpool(skip) if self.down_size > 1 else skip
        else:
            skip = self.avgpool(x) if self.down_size > 1 else x
            skip = self.conv1(skip)


        x = self.batchnorm(x) if self.use_bn else x
        x = self.leakyrelu(x) if self.pre_activation else x
        x = self.conv2(x)

        x = self.batchnorm(x) if self.use_bn else x
        x = self.leakyrelu(x) if self.pre_activation else x
        x = self.conv3(x)

        x = self.avgpool(x) if self.down_size > 1 else x
        x = torch.add(x, skip)

        return x

class SpadeBN(nn.Module):
    def __init__(self, x_in_channel, m_in_channel, x_hidden, spade_hidden, m_up_size, kernel_size=(3,3)):
        super().__init__()

        self.spade_hidden = spade_hidden

        self.batchnorm = nn.BatchNorm2d(x_in_channel, affine=False)
        self.upsample = nn.Upsample(scale_factor=m_up_size, mode="nearest")
        if spade_hidden != False:
            self.conv1 = nn.Conv2d(m_in_channel, spade_hidden, kernel_size, padding="same")
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.0)
        self.conv2 = nn.Conv2d(m_in_channel, x_hidden, kernel_size, padding="same")
        self.conv3 = nn.Conv2d(m_in_channel, x_hidden, kernel_size, padding="same")

    def forward(self, x, m):
        x = self.batchnorm(x)
        if self.spade_hidden != False:
            m = self.upsample(m)
            m = self.conv1(m)
            m = self.leakyrelu(m)

        gamma = self.conv2(m)
        beta = self.conv3(m)
        out = x * (1 + gamma) + beta

        return out

class ResblockUpSpadeSR(nn.Module):
    def __init__(self, x_in_channel, m1_in_channel, m2_in_channel, x_hidden, spade_hidden, spade_up_size, in_channels, kernel_size=(3,3), up_size=2, pre_activation=True, neg_slope=0.0):
        super().__init__()

        self.up_size = up_size
        self.pre_activation = pre_activation

        self.upsample = nn.Upsample(scale_factor=up_size, mode="nearest")
        self.conv1 = nn.Conv2d(x_in_channel, x_hidden, (1, 1), stride=(1,1), padding="same", bias=False)
        
        self.spadeBN1 = SpadeBN(x_in_channel, m1_in_channel, in_channels, False, None)
        self.spadeBN2 = SpadeBN(x_hidden, m2_in_channel, x_hidden, False, None)
        self.leakyrelu = nn.LeakyReLU(negative_slope=neg_slope)

        self.conv2 = nn.Conv2d(in_channels, x_hidden, kernel_size, padding="same", bias=False)
        self.conv3 = nn.Conv2d(x_hidden, x_hidden, kernel_size, padding="same", bias=False)

    def forward(self, x, m1, m2):
        skip = self.upsample(x) if self.up_size > 1 else x
        skip = self.conv1(skip)

        x = self.spadeBN1(x, m1)
        x = self.leakyrelu(x) if self.pre_activation else x
        x = self.upsample(x) if self.up_size > 1 else x
        x = self.conv2(x)

        x = self.spadeBN2(x, m2)
        x = self.leakyrelu(x)
        x = self.conv3(x)

        x = torch.add(x, skip)

        return x


if __name__ == "__main__":

    net = ResblockDown(1, 128, pre_activation=False, use_bn=False, neg_slope= 0.01, down_size=1)
    ins = torch.randn(1, 1, 5, 8)
    out = net(ins)
    print(out.shape)

