import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2

class M_cov(nn.Module):
    def __init__(self, in_channels, out_channels):
        super (M_cov, self).__init__()
        self.out_channels = out_channels
        # xin_channels = torch.cat(mo)
        self.fusecov = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False,groups=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out = self.fusecov(x)
        return out

class MFIAM(nn.Module):
    def __init__(self, mobile_channel, vit_channel, dim):
        super(MFIAM, self).__init__()
        self.mobile_channel = mobile_channel
        self.vit_channel = vit_channel
        self.mobile_ca = CA(mobile_channel, mobile_channel)
        self.vit_ca = CA(vit_channel, vit_channel)
        self.sc = FeatureRectifyModule(dim=dim)
        self.q_ocovn = nn.Conv2d(vit_channel, mobile_channel,kernel_size=1,bias=False)
        self.h_ocovn = nn.Conv2d(mobile_channel, vit_channel,kernel_size=1,bias=False)
    def forward(self,m_x,v_y):
        m_xca = self.mobile_ca(m_x)
        v_yca = self.vit_ca(v_y)
        v_yca = self.q_ocovn(v_yca)
        m_xca, v_yca = self.sc(m_xca, v_yca)
        v_yca = self.h_ocovn(v_yca)
        return m_xca,v_yca

class MOBILE(nn.Module):
    def __init__(self,fout_channels, dimesin = 256):
        super(MOBILE, self).__init__()
        self.fout_channels = fout_channels
        self.mobile_ca = CA(fout_channels, fout_channels)
        self.mobile = nn.Sequential(nn.Conv2d(fout_channels,dimesin,kernel_size=7,bias=False),
            nn.BatchNorm2d(dimesin,affine=False))
    def forward(self,x):
        x = self.mobile_ca(x)
        x = self.mobile(x)
        x = x.view(x.size(0), -1)
        return desc_l2norm(x)
class MVIT(nn.Module):
    def __init__(self,mvit_channels, mvit_dimesin = 256):
        super(MVIT, self).__init__()
        self.mvit_channels = mvit_channels
        self.mvit_ca = CA(mvit_channels, mvit_channels)
        self.mvit = nn.Sequential(nn.Conv2d(mvit_channels,mvit_dimesin,kernel_size=7,bias=False),
            nn.BatchNorm2d(mvit_dimesin,affine=False))
    def forward(self,y):
        y = self.mvit_ca(y)
        y = self.mvit(y)
        y = y.view(y.size(0), -1)
        return desc_l2norm(y)
def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    eps_l2_norm = 1e-10
    desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)
    return desc

class DFFM(nn.Module):
    def __init__(self,m_channels,v_channels,fout_channels, dimesin = 256):
        super(DFFM, self).__init__()
        self.fout_channels = fout_channels
        self.om_conv = nn.Conv2d(m_channels,fout_channels,kernel_size=1,bias=False)
        self.ov_conv = nn.Conv2d(v_channels,fout_channels,kernel_size=1,bias=False)
        self.fuse_cov = M_cov(m_channels+v_channels, fout_channels)
        self.norm = nn.BatchNorm2d(fout_channels)
        self.ca = CA(fout_channels,fout_channels)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(fout_channels,dimesin,kernel_size=7,bias=False),
            nn.BatchNorm2d(dimesin,affine=False)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self,m_x,v_y):
        o_x = m_x
        o_y = v_y
        m_x = self.om_conv(m_x)
        v_y = self.ov_conv(v_y)
        xy_cat = torch.cat((o_x, o_y),dim=1)
        xy_cat1 = self.fuse_cov(xy_cat)
        xy_cat = torch.cat((xy_cat1, xy_cat1),dim=1)
        weight = self.softmax(xy_cat)
        weight = torch.split(weight, self.fout_channels, dim=1)
        m_xca = weight[0]*m_x
        v_yca = weight[1]*v_y
        out = m_xca+v_yca
        out = self.norm(out)
        out = self.ca(out)
        out = self.Conv3(out)
        return out
