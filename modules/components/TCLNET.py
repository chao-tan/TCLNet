from torch import nn
#################################################################
############ tools and components ###############################

class ConvBlock(nn.Module):
    # a set of conv-bn-relu operation
    def __init__(self, inp_dim, out_dim, kernel_size, stride, use_bn, use_relu):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = ConvBlock(in_dim,int(out_dim/2),1,stride=1,use_bn=False,use_relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = ConvBlock(int(out_dim/2),int(out_dim/2),3,stride=1,use_bn=False,use_relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = ConvBlock(int(out_dim/2),out_dim,1,stride=1,use_bn=False,use_relu=False)
        self.expand_conv = None
        if in_dim != out_dim:
            self.expand_conv =ConvBlock(in_dim,out_dim,1,1,False,False)


    def forward(self, x):
        if self.expand_conv is not None:
            residual = self.expand_conv(x)
        else: residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        return out


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.maxpooling = nn.MaxPool2d(2,2)
        self.upsampling = nn.Upsample(scale_factor=2)

        self.pre = nn.Sequential( ConvBlock(3, 16, 7, 2, use_bn=True, use_relu=True),
                                  ConvBlock(16,32, 1,1,use_bn=False,use_relu=False),
                                  ResBlock(32,32),
                                  nn.MaxPool2d(2, 2),
                                  ResBlock(32,32),
                                  ConvBlock(32,64,1, 1, use_bn=False, use_relu=False),
                                  ResBlock(64,64))

        self.down1 = ResBlock(64,128)
        self.down2 = ResBlock(128,256)
        self.down3 = ResBlock(256,256)

        self.inner = ResBlock(256,256)

        self.up3 = ResBlock(256,256)
        self.up2 = ResBlock(256,128)
        self.up1 = ResBlock(128,64)

        self.outter = nn.Sequential(ResBlock(64,64),
                                    ConvBlock(64, 64, 1, 1, use_bn=True, use_relu=True),
                                    ConvBlock(64, 1, 1, 1, use_bn=False, use_relu=False))


    def forward(self, x):
        x = self.pre(x)

        x = self.maxpooling(self.down1(x))
        x = self.maxpooling(self.down2(x))
        x = self.maxpooling(self.down3(x))

        x = self.inner(x)

        x = self.upsampling(self.up3(x))
        x = self.upsampling(self.up2(x))
        x = self.upsampling(self.up1(x))

        x = self.outter(x)

        return x