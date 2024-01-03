import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Head import BiMFFHead,CA_BIMFFHead
from timm.models.layers import DropPath, to_2tuple
from .DFF import DiffNet
from .base_decoder import Base_Decoder
from .Att import *


class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp,
                 kv_per_win,kv_downsample_ratio,kv_downsample_kernel,kv_downsample_mode,
                 topk,param_attention,param_routing,diff_routing, soft_routing,side_dwconv,
                 auto_pad
                 ):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop)
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'B':
                self.attns.append(
                    BiLevelRoutingAttention(dim=dim_embed, num_heads=heads, n_win=window_size, qk_dim=dim_embed,
                                        kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x)

        positions = []
        references = []
        for d in range(self.depths):
            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references

class Conv_BiMFF(nn.Module):
    def __init__(self, in_channels=[128, 256, 512, 1024], mla_channels=128):
        super(Conv_BiMFF, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(in_channels[0], mla_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(in_channels[1], mla_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(in_channels[2], mla_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(in_channels[3], mla_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

        self.mla_b2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_b3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_b4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_b5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.upscore5p = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore4p = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore3p = nn.Upsample(scale_factor=2, mode='bilinear')

        self.downcore2p = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.downcore3p = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.downcore4p = nn.MaxPool2d(2, 2, ceil_mode=True)

    def to_2D(self, x):
        n, hw, c = x.shape
        h=w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x

    def forward(self, res2, res3, res4, res5):
        if len(res2.size()) == 3:
            res2 = self.to_2D(res2)  #[1, 96, 56, 56]
            res3 = self.to_2D(res3)  #[1, 192, 28, 28]
            res4 = self.to_2D(res4)  #[1, 384, 14, 14]
            res5 = self.to_2D(res5)  #[1, 768, 7, 7]
        #p代表从5-4-3-2自底向上传递，b代表2，3，4，5自顶向下传递


        mla_p5_1x1 = self.mla_p5_1x1(res5)#1 96 7 7
        mla_p4_1x1 = self.mla_p4_1x1(res4)#1 96 14 14
        mla_p3_1x1 = self.mla_p3_1x1(res3)#1 96 28 28
        mla_p2_1x1 = self.mla_p2_1x1(res2)#1 96 56 56

        mla_p4_plus = self.upscore5p(mla_p5_1x1) + mla_p4_1x1 #1 96 14 14
        mla_p3_plus = self.upscore4p(mla_p4_plus) + mla_p3_1x1 #1 96 28 28
        mla_p2_plus = self.upscore3p(mla_p3_plus) + mla_p2_1x1 #1 96 56 56

        mla_p5 = self.mla_p5(mla_p5_1x1)#1 96 7 7
        mla_p4 = self.mla_p4(mla_p4_plus)#1 96 14 14
        mla_p3 = self.mla_p3(mla_p3_plus)#1 96 28 28
        mla_p2 = self.mla_p2(mla_p2_plus)#1 96 56 56

        mla_b2_plus = mla_p2_1x1 #1 96 56 56
        mla_b3_plus = self.downcore2p(mla_b2_plus) + mla_p3_1x1#1 96 28 28
        mla_b4_plus = self.downcore2p(mla_b3_plus) + mla_p4_1x1#1 96 14 14
        mla_b5_plus = self.downcore2p(mla_b4_plus) + mla_p5_1x1#1 96 7 7

        mla_b2 = self.mla_b2(mla_b2_plus)#1 96 56 56
        mla_b3 = self.mla_b3(mla_b3_plus)#1 96 28 28
        mla_b4 = self.mla_b4(mla_b4_plus)#1 96 14 14
        mla_b5 = self.mla_b5(mla_b5_plus)#1 96 7 7

        mla_b2 = torch.cat((mla_b2, mla_p2),dim=1)#1 192 56 56
        mla_b3 = torch.cat((mla_b3, mla_p3),dim=1)#1 192 28 28
        mla_b4 = torch.cat((mla_b4, mla_p4),dim=1)#1 192 14 14
        mla_b5 = torch.cat((mla_b5, mla_p5),dim=1)#1 192 7 7

        return mla_b2, mla_b3, mla_b4, mla_b5


class CA_Net(nn.Module):

    def __init__(self, img_size=384, patch_size=4, num_classes=2, expansion=4,
                 dim_stem=128, dims=[128, 256, 512, 1024], depths=[2, 2, 12, 2],
                 heads=[4, 8, 16, 32],
                 window_sizes=[12, 12, 12, 12],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.5,
                 strides=[-1, -1, 1, 1], offset_range_factor=[-1, -1, 2, 2],
                 # strides=[1, 1, 1, 1], offset_range_factor=[1, 1, 2, 2],
                 stage_spec=[['L', 'S'], ['L', 'S'], ['D', 'D', 'B',  'D', 'D', 'B',  'D', 'D', 'B',  'D', 'D', 'B'], ['L', 'S']],
                 # stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'S','L', 'S','L', 'S','L', 'S','L', 'S','L', 'S'],['L', 'S']],
                 # stage_spec=[['B', 'D'], ['B', 'D'], ['B', 'D','B', 'D','B', 'D','B', 'D','B', 'D','B', 'D'],['B', 'D']],
                 groups=[-1, -1, 4, 8],
                 # groups=[1, 1, 4, 8],
                 use_pes=[False, False, True, True],
                 dwc_pes=[False, False, False, False],
                 # sr_ratios=[-1, -1, -1, -1],
                 sr_ratios=[1, 1, 1, 1],
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 kv_per_wins=[2, 2, 1, 1],kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],kv_downsample_mode='ada_avgpool',
                 topks=[8, 8, 1, 1],param_attention='qkvo',
                 param_routing=False, diff_routing=False, soft_routing=False,side_dwconv=5,
                 auto_pad=False,model_choice = 'CAN',
                 **kwargs):
        super().__init__()
        # self.out_indices = [1,3,5,7,9,11]
        self.model_choice = model_choice
        self.mla = Conv_BiMFF(in_channels=dims,mla_channels=dims[0])
        self.bi_head = CA_BIMFFHead(mla_channels=dims[1],num_classes=num_classes)
        self.patch_proj = nn.Sequential(
            nn.Conv2d(5, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(5, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.uplayers = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        for i in range(len(depths)):
            # dim1 96 *2 *2 *2 dim2 96,192,384,768
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            # fmap_size, window_size, ns_per_pt,
            # dim_in, dim_embed, depths, stage_spec, n_groups,
            # use_pe, sr_ratio,
            # heads, stride, offset_range_factor, stage_idx,
            # dwc_pe, no_off, fixed_pe,
            # attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp
            self.stages.append(
                TransformerStage(fmap_size=img_size,window_size=window_sizes[i], ns_per_pt=ns_per_pts[i],
                                 dim_in=dim1, dim_embed=dims[i], depths=depths[i], stage_spec=stage_spec[i],
                                 n_groups=groups[i],use_pe=use_pes[i],sr_ratio=sr_ratios[i], heads=heads[i],
                                 stride=strides[i],offset_range_factor=offset_range_factor[i], stage_idx=i,
                                 dwc_pe=dwc_pes[i], no_off=no_offs[i], fixed_pe=fixed_pes[i],
                                 attn_drop=attn_drop_rate, proj_drop=drop_rate, expansion=expansion,
                                 drop=drop_rate,drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 use_dwc_mlp=use_dwc_mlps[i],kv_per_win=kv_per_wins[i],kv_downsample_ratio=kv_downsample_ratios[i],
                                 kv_downsample_kernel=kv_downsample_kernels[i],kv_downsample_mode=kv_downsample_mode,
                                 topk=topks[i],param_attention=param_attention,
                                 param_routing=param_routing, diff_routing=diff_routing, soft_routing=soft_routing,
                                 side_dwconv=side_dwconv,auto_pad=auto_pad)
            )
            self.norm_list.append(LayerNormProxy(dim1))
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
        self.cls_norm = LayerNormProxy(dims[-1])
        self.dff = DiffNet(1,64,num_classes=num_classes)
        # self.cls_head = nn.Linear(dims[-1], num_classes)
        self.decoder = Base_Decoder(num_classes)

        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict):

        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)

        self.load_state_dict(new_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        #C 128, 256, 512, 1024
        #H,W  56 28 14 7
        # B C H W  -> B emb_dim H/4 W/4
        x = self.patch_proj(x)

        positions = []
        references = []
        outs = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
            outs.append(x)

            if i < 3:
                x = self.down_projs[i](x)
        #base decoder
        if self.model_choice == 'BASE':
            x = self.decoder(x)
            return  x
        if self.model_choice == 'BI':
            c1 = self.norm_list[0](outs[0])  # out_indices = [1,3,9,11]
            c3 = self.norm_list[1](outs[1])
            c9 = self.norm_list[2](outs[2])
            c11 = self.norm_list[3](outs[3])
            #[B, 196, 56,28,14,7,...]
            d2,d3,d4,d5 = self.mla(c1, c3, c9, c11)
                # B num_classes 224 224
            edge,d2,d3,d4,d5 = self.bi_head((d2,d3,d4,d5))
            return F.sigmoid(edge),F.sigmoid(d2),F.sigmoid(d3),F.sigmoid(d4),F.sigmoid(d5)
        if self.model_choice == 'CAN':
            c1 = self.norm_list[0](outs[0])  # out_indices = [1,3,9,11]
            c3 = self.norm_list[1](outs[1])
            c9 = self.norm_list[2](outs[2])
            c11 = self.norm_list[3](outs[3])
            #[B, 196, 56,28,14,7,...]
            d2,d3,d4,d5 = self.mla(c1, c3, c9, c11)
                # B num_classes 224 224
            edge,d2,d3,d4,d5 = self.bi_head((d2,d3,d4,d5))

            d1 = self.dff(edge)

            return F.sigmoid(d1),F.sigmoid(d2),F.sigmoid(d3),F.sigmoid(d4),F.sigmoid(d5)
            # return d1,d2,d3,d4,d5
# if __name__ == '__main__':
#     model = CA_Net()
#     print(model)