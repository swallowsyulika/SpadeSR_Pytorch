import torch
import torch.nn as nn
from blocks_v2 import ResblockUp, ResblockDown, SpadeBN, ResblockUpSpadeSR
import yaml

class GeneratorSpadeSR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # cfg[in_channel] need setting
        self.res_up_c1 = ResblockUp(cfg['G_IN_CH'], cfg['G_CONV_THERMAL_CH'], pre_activation=False, use_bn=False, neg_slope=cfg['RELU_NEG_SLOPE'], up_size=1)
        self.res_up_c2 = ResblockUp(cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_THERMAL_CH'], neg_slope=cfg['RELU_NEG_SLOPE'], use_bn=False)
        self.res_up_c3 = ResblockUp(cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_THERMAL_CH'], neg_slope=cfg['RELU_NEG_SLOPE'], use_bn=False)
        self.res_up_c4 = ResblockUp(cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_THERMAL_CH'], neg_slope=cfg['RELU_NEG_SLOPE'], use_bn=False)
        
        self.liner = nn.Linear(cfg['G_Z_DIM'], 5*8*cfg['G_CONV_CH']*4, bias=False)

        self.leakyrelu = nn.LeakyReLU(negative_slope=cfg['RELU_NEG_SLOPE'])

        self.res_spsr_1 = ResblockUpSpadeSR(cfg['G_CONV_CH']*4, cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_CH']*4, False, None, in_channels=cfg['G_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.res_spsr_2 = ResblockUpSpadeSR(cfg['G_CONV_CH']*4, cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_CH']*2, False, None, in_channels=cfg['G_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE'])
        self.res_spsr_3 = ResblockUpSpadeSR(cfg['G_CONV_CH']*2, cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_CH'], False, None, in_channels=cfg['G_CONV_CH']*2, neg_slope=cfg['RELU_NEG_SLOPE'])
        
        self.spadeBN = SpadeBN(cfg['G_CONV_CH'], cfg['G_CONV_THERMAL_CH'], cfg['G_CONV_CH'], False, None)

        self.conv = nn.Conv2d(cfg['G_CONV_CH'], 3, (3, 3), padding="same", bias=True)
        self.tanh = nn.Tanh()


    def forward(self, thermal_input, noise):

        thermal_c1 = self.res_up_c1(thermal_input)
        thermal_c2 = self.res_up_c2(thermal_c1)
        thermal_c3 = self.res_up_c3(thermal_c2)
        thermal_c4 = self.res_up_c4(thermal_c3)

        ### change if input size not correct
        x = self.liner(noise)
        x = x.view((-1, self.cfg['G_CONV_CH']*4, 5, 8))
        ###

        c1_relu = self.leakyrelu(thermal_c1)
        c2_relu = self.leakyrelu(thermal_c2)
        c3_relu = self.leakyrelu(thermal_c3)
        c4_relu = self.leakyrelu(thermal_c4)
        
        x = self.res_spsr_1(x, c1_relu, c2_relu)
        x = self.res_spsr_2(x, c2_relu, c3_relu)
        x = self.res_spsr_3(x, c3_relu, c4_relu)

        x = self.spadeBN(x, c4_relu)
        x = self.leakyrelu(x)

        out_img = self.conv(x)
        out_img = self.tanh(out_img)

        return out_img


if __name__ == "__main__":
    with open("./cfgs/SPADE_SR_64.yaml", "r") as stream:
        cfg = yaml.load(stream)

    thermal_input  =  torch.randn(200, 1, 5, 8)
    noise = torch.randn(cfg['G_Z_DIM'])

    net = GeneratorSpadeSR(cfg)
    out = net(thermal_input, noise)
    print(out.shape)