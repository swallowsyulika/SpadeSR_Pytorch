import torch
import torch.nn as nn
from blocks_v2 import ResblockUp, ResblockDown
import yaml


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.res_down_c1 = ResblockDown(cfg['D_IN_CH'], cfg['D_CONV_CH'], pre_activation=False, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.res_down_c2 = ResblockDown(cfg['D_CONV_CH'], cfg['D_CONV_CH']*2, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.res_down_c3 = ResblockDown(cfg['D_CONV_CH']*2, cfg['D_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE'])

        self.leakyrelu = nn.LeakyReLU(negative_slope=cfg['RELU_NEG_SLOPE'])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(cfg['D_CONV_CH']*4*5*8, 1)

        self.res_down_th = ResblockDown(cfg['D_CONV_CH']*4, cfg['D_CONV_THERMAL_CH'], down_size=1, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.conv_th = nn.Conv2d(cfg['D_CONV_THERMAL_CH'], 1, (3,3), padding='same')
        self.tanh = nn.Tanh()

    def forward(self, x):
        c1 = self.res_down_c1(x)
        c2 = self.res_down_c2(c1)
        c3 = self.res_down_c3(c2)

        feature = self.leakyrelu(c3)

        if self.cfg['D_USE_GSP']:
            # need check
            feature = torch.sum(feature, axis=[1, 2])
        else:
            feature = self.flatten(feature)

        label = self.linear(feature)
        thermal_rec = self.res_down_th(c3)
        thermal_rec = self.leakyrelu(thermal_rec)
        thermal_rec = self.conv_th(thermal_rec)
        thermal_rec = self.tanh(thermal_rec)
        
        return label, thermal_rec


if __name__ == "__main__":
    with open("./cfgs/SPADE_SR_64.yaml", "r") as stream:
        cfg = yaml.load(stream)

    inputs = torch.randn(256, 3, 40, 64)
    net = Discriminator(cfg)

    label, out = net(inputs)
    print(label.shape, out.shape)