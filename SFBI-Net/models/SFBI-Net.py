from torch import nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import sys
sys.path.append('./')
from models.resnet import resnet18,resnet34,resnet50
from models.help_funcs import TwoLayerConv2d
from torch.nn import functional as F

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
        self.apply(self._init_weights)

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

    def forward(self, x,h,w):
        x = self.fc1(x)

        x = self.dwconv(x,h,w)
        x = self.act(x)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class CrossGlobalLocalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.kv = nn.Linear(dim,dim*2,bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.local = local_attention(dim)
        
        self.apply(self._init_weights) # init

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

    def _get_attn(self, x):

        B,N,C = x.size()

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        
        return q,kv[0],kv[1]
    def forward(self,x1,x2,h,w):
        
        b,n,c = x1.size()
       
        q_1,k_1,v_1 = self._get_attn(x1)
        q_2,k_2,v_2 = self._get_attn(x2)
       
        dots_1 = (q_2@k_1.transpose(-2,-1))*self.scale
        dots_2 = (q_1@k_2.transpose(-2,-1))*self.scale
       
        attn_1 = dots_1.softmax(dim=-1)
        attn_2 = dots_2.softmax(dim=-1)

        global1 = self.proj_drop(self.proj((attn_1@v_1).transpose(1,2).reshape(b,-1,c)))
        global2 = self.proj_drop(self.proj((attn_2@v_2).transpose(1,2).reshape(b,-1,c)))
        
        local1,local2 = self.local(x1,x2,h,w)

        return global1+local1,global2+local2 

class local_attention(nn.Module):
    r'''
    local attention
    ''' 
    def __init__(self,dim) -> None:
        super().__init__()
        self.local1 = ConvBN(dim, dim, kernel_size=1)
        self.local2 = ConvBN(dim, dim, kernel_size=3)
        
    def forward(self, x1,x2,h,w):
        b,_,c = x1.size()
        x1 = x1.reshape(b,-1,h,w)
        x2 = x2.reshape(b,-1,h,w)
        
        local1 = self.local1(x1) + self.local2(x1)
        local2 = self.local1(x2) + self.local2(x2)
        return local1.reshape(b,-1,c),local2.reshape(b,-1,c)
 
class CGLA_Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossGlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)


    def forward(self, x1,x2):
        b,c,h,w = x1.size()
        x1 = x1.reshape(b,-1,c) #  b,c,h,w -> b,h*w,c
        x2 = x2.reshape(b,-1,c)
        outs = self.drop_path(self.attn(self.norm1(x1), self.norm1(x2),h,w)) # x1,x2
        x1 = x1.reshape(b,-1,c) + outs[0]
        x2 = x2.reshape(b,-1,c) + outs[1]
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1),h,w))
        x2 = x2 + self.drop_path(self.mlp(self.norm2(x2),h,w))
        return x1.reshape(b,c,h,w),x2.reshape(b,c,h,w)
    


class SFD_Loss(nn.Module):        
    def __init__(self,dim,class_num):
        super(SFD_Loss, self).__init__()
        self.class_num = class_num
        self.OneClassChannelNum = int(dim/class_num) 
        self.kernel_size =  (1,self.OneClassChannelNum)
        self.stride = self.kernel_size
        self.max_pooling = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
    def forward(self,x):
        # diverse loss
        # reshape b,c,h,w -> b,c,hw
        branch = x.reshape(x.size(0),x.size(1), x.size(2) * x.size(3))
        branch_q = branch
        branch_k = branch.permute(0,2,1)
        branch_v = branch
        att = F.softmax(torch.bmm(branch_q,branch_k),dim=2)
        branch = torch.bmm(att,branch_v)
        # softmax on spatial dim
        branch = F.softmax(branch,2)
        # b,c,hw -> b,c,h,w
        branch = branch.reshape(x.size(0),x.size(2), x.size(3), x.size(1))
        # Cross-channel Max Pooling calculates the max value of different class
        branch = self.max_pooling(branch)
        branch = branch.permute(0,3,1,2)
        branch = F.max_pool2d(branch, kernel_size=3, stride=1, padding=1)
        # b,c,h,w -> b,c,hw
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        # avg of each class
        sfd_loss = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/self.OneClassChannelNum
        return sfd_loss
    



class backboneEncoder(nn.Module):
    def __init__(self, 
                resnet_stages_num=5, backbone='resnet18',dim=None,num_heads=8):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(backboneEncoder, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True]) # 使用空洞卷积代替步幅卷积，增加模型感受野
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError

        ########### cross attention ###########
        self.cross_attention_1 =  CGLA_Block(dim=dim[0],num_heads=num_heads)
        self.cross_attention_2 =  CGLA_Block(dim=dim[1],num_heads=num_heads)
        self.cross_attention_3 =  CGLA_Block(dim=dim[2],num_heads=num_heads)
        self.cross_attention_4 =  CGLA_Block(dim=dim[3],num_heads=num_heads)
        ########################################
        
        self.resnet_stream = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.resnet_stages_num = resnet_stages_num

    def forward(self, x1, x2):
        x1,x2 = self.forward_encoder(x1,x2)

        return x1,x2

    def forward_encoder(self, x1,x2):

        x1 = self.resnet_stream(x1) 
        x2 = self.resnet_stream(x2)
        x1_4 = self.resnet.layer1(x1) # 1/4, in=64, out=64
        x2_4 = self.resnet.layer1(x2) # 1/4, in=64, out=64
        x1_4,x2_4 = self.cross_attention_1(x1_4,x2_4)
        x1_8 = self.resnet.layer2(x1_4) # 1/8, in=64, out=128
        x2_8 = self.resnet.layer2(x2_4) # 1/8, in=64, out=128
        x1_8,x2_8 = self.cross_attention_2(x1_8,x2_8)
        if self.resnet_stages_num > 3:
            x1_8 = self.resnet.layer3(x1_8) # 1/8, in=128, out=256
            x2_8 = self.resnet.layer3(x2_8) # 1/8, in=128, out=256
            x1_8,x2_8 = self.cross_attention_3(x1_8,x2_8)
        if self.resnet_stages_num == 5:
            x1_8 = self.resnet.layer4(x1_8) # 1/32, in=256, out=512
            x2_8 = self.resnet.layer4(x2_8) # 1/32, in=256, out=512
            x1_8,x2_8 = self.cross_attention_4(x1_8,x2_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        return x1_8,x2_8

class FeatureFusion(nn.Module):
    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices

    @staticmethod
    def fusion(x1, x2,policy):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x

    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


class decoder(nn.Module):  
    def __init__(self,fusion_policy,class_n,output_sigmoid) -> None:
        super().__init__()

        self.upscalex2 = nn.Upsample(scale_factor=2)
        self.upscalex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_pred = nn.Conv2d(512, 32, kernel_size=3, padding=1) 
        self.neck_layer = FeatureFusion(fusion_policy)
        self.sfd = SFD_Loss(dim=32*2,class_num=2)
        self.cls_head = TwoLayerConv2d(in_channels=32*2 if fusion_policy == "concat" else 32, out_channels=class_n)
        self.sigmoid = nn.Sigmoid()
        self.output_sigmoid = output_sigmoid
    def forward(self, x1,x2):
        x1 = self.upscalex2(x1)
        x2 = self.upscalex2(x2)

        x1 = self.conv_pred(x1)
        x2 = self.conv_pred(x2)
        # fusion
        x = self.neck_layer.fusion(x1, x2,self.neck_layer.policy)

        if self.training:
            SpatialFocusDiveristy_loss = self.sfd(x)

        x = self.upscalex4(x)

        out = self.cls_head(x)
        if self.output_sigmoid:
            out = self.sigmoid(out)
        if self.training:
            return [out,SpatialFocusDiveristy_loss]
        else:
            return [out]

class SFBIN(nn.Module):
    def __init__(self,class_n=2,backbone='resnet18',output_sigmoid=False,
                 resnet_stages_num=5, dim=[64,128,256,512],num_heads=8,fusion_policy='concat'):
        super(SFBIN,self).__init__()
        self.encoder = backboneEncoder(resnet_stages_num, backbone,dim,num_heads)
        self.decoder = decoder(fusion_policy,class_n, output_sigmoid)
        
    def forward(self, x1,x2):
        x1,x2 = self.encoder(x1,x2)
        x = self.decoder(x1,x2)
        return x
    
if __name__ =='__main__':
    model = SFBIN(backbone='resnet18')
    x = torch.randn(1,3,256,256)
    y = torch.randn(1,3,256,256)
    out = model(x,y)
    print(out[0].shape)
