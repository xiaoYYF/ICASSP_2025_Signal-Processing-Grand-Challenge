"""
modification made on the basis of link:https://github.com/Xiaoccer/MobileFaceNet_Pytorch
"""
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torchaudio


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

# original mobilefacenet setting
# Mobilefacenet_bottleneck_setting = [
#     # t, c , n ,s
#     [2, 64, 5, 2],
#     [4, 128, 1, 2],
#     [2, 128, 6, 1],
#     [4, 128, 1, 2],
#     [2, 128, 2, 1]
# ]

# Mobilenetv2_bottleneck_setting = [
#     # t, c, n, s
#     [1, 16, 1, 1],
#     [6, 24, 2, 2],
#     [6, 32, 3, 2],
#     [6, 64, 4, 2],
#     [6, 96, 3, 1],
#     [6, 160, 3, 2],
#     [6, 320, 1, 1],
# ]

# refer to DCASE2022 Task2 Top-1
# https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        # self.conv1 = ConvBlock(2, 64, 3, 2, 1)
        # 将输入通道数改为 6
        self.conv1 = ConvBlock(6, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 5), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(256, 2800)




        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)

        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)

        return out,feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=256, hop_len=35):
        super(TgramNet, self).__init__()
        # 将输入通道数依旧为1，分开输入再拼接。
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(81),  # LayerNorm 针对最后一维是 313 的输入
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)]
        )

    def forward(self, x):
        # 期望 x 的形状为 [batch_size, channels=3, time_steps]
        # print(f"Shape of x:{x.shape}") # Shape of x:torch.Size([32, 3, 4608])
        out = self.conv_extractor(x)
        out = self.conv_encoder(out)

        return out


class STgramMFN(nn.Module):
    def __init__(self, num_classes, c_dim=128, win_len=256, hop_len=35, bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=False, m=0.5, s=30, sub=1):
        super(STgramMFN, self).__init__()
        self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes, bottleneck_setting=bottleneck_setting)

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, label=None):
        # print("x_xwav shape:", x_xwav.shape)
        # print("x_xmel shape:", x_xmel.shape)
        # 分别处理 x_xwav, x_ywav, x_zwav 得到 TgramNet 提取的特征 x_xt, x_yt, x_zt
        x_xt = self.tgramnet(x_xwav.unsqueeze(1))  # 输入形状：[batch_size, 1, time_steps]
        x_yt = self.tgramnet(x_ywav.unsqueeze(1))  # 输入形状：[batch_size, 1, time_steps]
        x_zt = self.tgramnet(x_zwav.unsqueeze(1))  # 输入形状：[batch_size, 1, time_steps]


        x_xt, x_yt, x_zt = x_xt.unsqueeze(1), x_yt.unsqueeze(1), x_zt.unsqueeze(1)
        x_xmel, x_ymel, x_zmel = x_xmel.unsqueeze(1), x_ymel.unsqueeze(1), x_ymel.unsqueeze(1)
        
        # 将 TgramNet 提取的特征拼接在一起，形成 x_t
        x_t = torch.cat((x_xt, x_yt, x_zt), dim=1)  # 形状：[batch_size, 3, mel_bins, time_frames]
        x_mel = torch.cat((x_xmel, x_ymel, x_zmel), dim = 1)
        # print("x_mel shape:", x_mel.shape)  # 预期：[batch_size, 3, mel_bins, time_frames]
        # print("x_t shape:", x_t.shape)      # 预期：[batch_size, 3, mel_bins, time_frames]
        
        # 拼接 x_mel 和 x_t，形成输入给 MobileFaceNet 的特征
        x = torch.cat((x_mel, x_t), dim=1)  # 拼接后形状：[batch_size, 6, mel_bins, time_frames]
        
        # print("x shape after concatenation:", x.shape)  # 应该是 [batch_size, 6, mel_bins, time_frames]
        
        # 通过 MobileFaceNet 处理拼接后的特征
        out, feature = self.mobilefacenet(x, label)
        
        # 如果使用 arcface，则计算 arcface 输出
        # if self.arcface:
        #     out = self.arcface(feature, label)
        return out, feature


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


if __name__ == '__main__':
    net = STgramMFN(num_classes=10)
    x_wav = torch.randn((2, 16000*2))
