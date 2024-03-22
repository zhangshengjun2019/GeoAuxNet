"""
SparseUNet  GeoAuxNet
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import faiss
import einops
import pointops

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.models.layers import trunc_normal_

from ..builder import MODELS
from ..utils import offset2batch, batch2offset
import time
import faiss.contrib.torch_utils

class PDBatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        context_channels=256,
        eps=1e-3,
        momentum=0.01,
        conditions=("ScanNet", "S3DIS", "SemanticKITTI"),
        decouple=True,
        adaptive=False,
        affine=True,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        self.affine = affine
        if self.decouple:
            self.bns = nn.ModuleList(
                [
                    nn.BatchNorm1d(
                        num_features=num_features,
                        eps=eps,
                        momentum=momentum,
                        affine=affine,
                    )
                    for _ in conditions
                ]
            )
        else:
            self.bn = nn.BatchNorm1d(
                num_features=num_features, eps=eps, momentum=momentum, affine=affine
            )
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, feat, condition=None, context=None):
        if self.decouple:
            assert condition in self.conditions
            bn = self.bns[self.conditions.index(condition)]
        else:
            bn = self.bn
        feat = bn(feat)
        if self.adaptive:
            assert context is not None
            shift, scale = self.modulation(context).chunk(2, dim=1)
            feat = feat * (1.0 + scale) + shift
        return feat

class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            # TODO remove norm after project
            self.proj_conv = spconv.SubMConv3d(
                in_channels, embed_channels, kernel_size=1, bias=False
            )
            self.proj_norm = norm_fn(embed_channels)

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        x, condition, context = x
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features, condition, context))

        if self.in_channels == self.embed_channels:
            residual = self.proj(residual)
        else:
            residual = residual.replace_feature(
                self.proj_norm(self.proj_conv(residual).features, condition, context)
            )
        out = out.replace_feature(out.features + residual.features)
        out = out.replace_feature(self.relu(out.features))
        return out, condition, context

class CrossAttention(nn.Module):
    def __init__(
        self, 
        point_channels,
        voxel_channels,
        out_channels,
        mid_channels,
        ):
        super().__init__()
        self.linear_q = nn.Linear(point_channels, mid_channels)
        self.linear_k = nn.Linear(voxel_channels, mid_channels)
        self.linear_v = nn.Linear(point_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(mid_channels, elementwise_affine=False)
        self.mid_channels = mid_channels
        self.proj = nn.Linear(voxel_channels+out_channels, voxel_channels)
    def forward(self, point_features, voxel_features):
        q = self.linear_q(point_features)
        k = self.linear_k(voxel_features)
        v = self.linear_v(point_features)
        w = self.softmax(torch.mm(k, q.t())/self.mid_channels**0.5)
        output = self.norm(torch.mm(w,v))
        return torch.cat((voxel_features, output), dim=-1)

class SPConvDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        indice_key,
        kernel_size=2,
        bias=False,
        norm_fn=None,
    ):
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))
        return out

class SPConvUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        indice_key,
        kernel_size=2,
        bias=False,
        norm_fn=None,
    ):
        super().__init__()
        self.conv = spconv.SparseInverseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))
        return out

class SPConvPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, norm_fn=None):
        super().__init__()
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
            indice_key="stem",
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x

        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))
        return out

class HyperNet(nn.Module):
    def __init__(self, voxel_channels, point_channels, position_channels=32, layer_latent_channels=128, norm_fn=None):
        super().__init__()
        # hypernet for weight
        self.weight_linear1 = nn.Linear(voxel_channels+position_channels, point_channels)
        self.weight_bn = norm_fn(point_channels)
        self.weight_relu = nn.ReLU(inplace=True)
        self.weight_linear2 = nn.Linear(point_channels, point_channels)
        # hypernet for bias
        self.embedding_linear = nn.Linear(3, position_channels)
        self.embedding_bn = norm_fn(position_channels)
        self.embedding_relu = nn.ReLU(position_channels)
        self.bias_linear1 = nn.Linear(voxel_channels + position_channels, point_channels)
        self.bias_bn = norm_fn(point_channels)
        self.bias_relu = nn.ReLU(inplace=True)
        self.bias_linear2 = nn.Linear(point_channels, point_channels)
        # layer latent code
        self.layer_latent_code = nn.Parameter(torch.randn(layer_latent_channels))
        self.latent_linear1 = nn.Linear(layer_latent_channels, point_channels)
        self.latent_relu = nn.ReLU(inplace=True)
        self.latent_linear2 = nn.Linear(point_channels, point_channels)
        self.latent_softmax = nn.Softmax(dim=0)
        # point mlp
        self.point_bn = norm_fn(point_channels)
        self.point_relu = nn.ReLU(inplace=True)

    def forward(self, x, pfo, condition, min_coords, grid_size, voxel_gpu_index):
        # print("start hyper")
        p, f, o = pfo

        
        point_offset = torch.cat((torch.tensor([0]).to(p.device), o), dim=0)
        voxel_offset = torch.cat((torch.tensor([0]).to(p.device), batch2offset(x.indices[:,0])), dim=0)
        # print("point_offset: ", point_offset)
        position_emb = []
        voxel_features = []
        for i in range(point_offset.shape[0]-1):
            if voxel_offset[i+1] - voxel_offset[i] == 0:
                continue
            if point_offset[i+1] - point_offset[i] == 0:
                continue 

            voxel_position = x.indices[voxel_offset[i]:voxel_offset[i+1], 1:] * grid_size + min_coords[i] + grid_size/2
            
            min = torch.min(p[point_offset[i]:point_offset[i+1]], dim=0, keepdim=True)[0]
            max = torch.max(p[point_offset[i]:point_offset[i+1]], dim=0, keepdim=True)[0]
            bool_min = torch.all(voxel_position >= min, 1)
            bool_max = torch.all(voxel_position <= max, 1)
            bool_idx = bool_min & bool_max
            if bool_idx.all() == False:
                bool_idx = bool_idx==bool_idx 
            # print("reset")
            voxel_gpu_index.reset()
            # print("train")
            # print(voxel_position[bool_idx].shape)
            voxel_gpu_index.train(voxel_position[bool_idx].float())
            # print("add")
            voxel_gpu_index.add(voxel_position[bool_idx].float())
            # print("search")
            # print(p[point_offset[i]:point_offset[i+1]].shape)
            _, idx = voxel_gpu_index.search(p[point_offset[i]:point_offset[i+1]], 1)
            # print("finish")
            idx = idx.squeeze(-1)
            relative_p = p[point_offset[i]:point_offset[i+1]] - voxel_position[bool_idx][idx]

            position_emb.append(relative_p)
            voxel_feature = x.features[voxel_offset[i]:voxel_offset[i+1]][bool_idx][idx]
            if voxel_feature.ndim == 1:
                voxel_features.append(voxel_feature.unsqueeze(0))
            else:
                voxel_features.append(voxel_feature)

        position_emb = torch.cat(position_emb, dim=0)
        voxel_features = torch.cat(voxel_features, dim=0)
        
        position_emb = self.embedding_relu(self.embedding_bn(self.embedding_linear(position_emb), condition))
        x_embed = torch.cat((voxel_features, position_emb), dim=-1)
        w = self.weight_linear2(self.weight_relu(self.weight_bn(self.weight_linear1(x_embed), condition)))
        b = self.bias_linear2(self.bias_relu(self.bias_bn(self.bias_linear1(x_embed), condition)))
        layer_latent_code = self.latent_softmax(self.latent_linear2(self.latent_relu(self.latent_linear1(self.layer_latent_code))))
        w = w*layer_latent_code
        b = b*layer_latent_code
        f = self.point_relu(self.point_bn(w*f+b, condition))
        pfo = [p, f, o]

        return pfo

class BasicMLP(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, use_res=False, norm_fn=None):
        super().__init__()
        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.relu = nn.ModuleList()
        self.layers = layers
        self.use_res = use_res
        for i in range(layers):
            if i == 0:
                self.linear.append(nn.Linear(in_channels, out_channels))
            else:
                self.linear.append(nn.Linear(out_channels, out_channels))
            self.norm.append(norm_fn(out_channels))
            self.relu.append(nn.ReLU(inplace=True))
        if use_res:
            self.res_linear = nn.Linear(in_channels, out_channels)
            self.res_norm = norm_fn(out_channels)
            self.res_relu = nn.ReLU(inplace=True)
    def forward(self, x, condition):
        if self.use_res:
            res = self.res_relu(self.res_norm(self.res_linear(x), condition))
        for i in range(self.layers):
            x = self.relu[i](self.norm[i](self.linear[i](x), condition))
        if self.use_res:
            return x + res
        else:
            return x

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16, norm_fn=None):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = norm_fn(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo, condition):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride + 1], o[0].item() // self.stride + 1
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride + 1
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous(), condition)
            )  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x), condition))  # (n, c)
        return [p, x, o]     

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None, previous_planes=None, norm_fn=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = BasicMLP(2 * in_planes, in_planes, layers=1, norm_fn=norm_fn)
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = BasicMLP(in_planes, out_planes, layers=1, norm_fn=norm_fn)
            self.linear2 = BasicMLP(previous_planes, out_planes, layers=1, norm_fn=norm_fn)
            
    def forward(self, pxo1, pxo2=None, condition=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_bb = x_b.sum(0, True) / cnt
                for i in range(len(self.linear2)):
                    if i == 1:
                        x_bb = self.linear2[i](x_bb, condition)
                    else:
                        x_bb = self.linear2[i](x_bb)

                x_b = torch.cat(
                    (x_b, x_bb.repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x, condition)
        else:
            
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            assert p1.shape[0] == x1.shape[0]
            x = self.linear1(x1, condition) + pointops.interpolation(
                p2, p1, self.linear2(x2, condition), o2, o1
            )
        return x
        
@MODELS.register_module("SpUNet-GeoAuxNet")
class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels,
        pointnet_in_channels,
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        zero_init=True,
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=False,
        pointnet_base_channels=32,
        pointnet_channels=(32, 64, 128, 256, 256, 128, 96, 96),
        pointnet_layers=(2, 2, 2, 4, 2, 2, 2, 2),
        stride=(4, 4, 4, 4),
        nsample=(24, 24, 24, 24),
        geo_pool_max_size=(32, 64, 128, 256),
        ca_out_channels=(16, 32, 64, 128),
        thresold=0.9,
        update_rate=0.1,
        grid_size=0.05,
        sensors=("RGB-D", "LiDAR")
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        assert len(pointnet_layers) == len(channels)
        assert len(stride) == len(layers) // 2
        assert len(stride) == len(nsample)
        assert len(stride) == len(geo_pool_max_size)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.cls_mode = cls_mode
        self.conditions = conditions
        self.zero_init = zero_init
        self.geo_pool_max_size = geo_pool_max_size
        self.thresold=thresold
        self.update_rate = update_rate
        self.min_coord = None
        self.grid_size = grid_size
        self.sensors = sensors
        

        norm_fn = partial(
            PDBatchNorm,
            eps=1e-3,
            momentum=0.01,
            conditions=conditions,
            context_channels=context_channels,
            decouple=norm_decouple,
            adaptive=norm_adaptive,
            affine=norm_affine,
        )
        block = BasicBlock

        
        self.conv_input = nn.ModuleList()
        self.point_input = nn.ModuleList()
        for i in range(len(sensors)):
            self.conv_input.append(
                SPConvPatchEmbedding(
                    in_channels, base_channels, kernel_size=5, norm_fn=norm_fn
                )
            )
            self.point_input.append(
                BasicMLP(
                    pointnet_in_channels, pointnet_base_channels, norm_fn=norm_fn
                )
            )

        self.point_dec_input = BasicMLP(
            pointnet_channels[len(stride)-1], pointnet_channels[len(stride)], norm_fn=norm_fn
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None
        self.ca = nn.ModuleList()

        point_enc_channels = pointnet_base_channels
        point_dec_channels = pointnet_channels[-1]
        self.point_down = nn.ModuleList()
        self.point_up = nn.ModuleList()
        self.point_enc = nn.ModuleList()
        self.point_dec = nn.ModuleList()
        self.hypernet = nn.ModuleList()

        self.geo_pool = []
        for _ in range(len(sensors)):
            self.geo_pool.append([])

        for s in range(self.num_stages):
            # build an empty feature bank
            for i in range(len(sensors)):
                self.geo_pool[i].append(None)

            # encode num_stages
            ## downsampling for voxels
            self.down.append(
                SPConvDown(
                    enc_channels,
                    channels[s],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{s + 1}",
                    norm_fn=norm_fn,
                )
            )
            ## hypernet for point
            self.hypernet.append(
                HyperNet(
                    voxel_channels=enc_channels,
                    point_channels=point_enc_channels,
                    norm_fn=norm_fn
                )
            )
            ## downsampling for points
            self.point_down.append(
                TransitionDown(
                    in_planes=point_enc_channels,
                    out_planes=pointnet_channels[s],
                    stride=stride[s],
                    nsample=nsample[s],
                    norm_fn=norm_fn
                )
            )
            ## cross attention between voxel features and point features
            self.ca.append(
                CrossAttention(
                    point_channels=pointnet_channels[s],
                    voxel_channels=channels[s],
                    out_channels=ca_out_channels[s],
                    mid_channels=pointnet_channels[s] // 2,
                )
            )
            ## encoder for voxels
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s]+ca_out_channels[s],
                                    channels[s]+ca_out_channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            ## encoder for points
            self.point_enc.append(
                BasicMLP(
                    pointnet_channels[s], 
                    pointnet_channels[s], 
                    layers=pointnet_layers[s], 
                    use_res=True, 
                    norm_fn=norm_fn
                )
            )

            if not self.cls_mode:
                # decode num_stages
                ## upsampling for voxels
                self.up.append(
                    SPConvUp(
                        channels[len(channels) - s - 1]+ca_out_channels[s] if s==self.num_stages-1 else channels[len(channels) - s - 1],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        norm_fn=norm_fn,
                    )
                )
                ## unsampling for points
                self.point_up.append(
                    TransitionUp(
                        in_planes=pointnet_channels[s-1] if s!=0 else pointnet_base_channels,                        
                        out_planes=point_dec_channels,
                        previous_planes=pointnet_channels[len(pointnet_channels) - s - 1],
                        norm_fn=norm_fn,
                    )
                )
                ## decoder for voxels
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    f"block{i}",
                                    block(
                                        dec_channels + enc_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                if i == 0
                                else (
                                    f"block{i}",
                                    block(
                                        dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}",
                                    ),
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )
                ## decoder for points
                self.point_dec.append(
                    BasicMLP(
                        point_dec_channels, 
                        point_dec_channels, 
                        layers=layers[len(pointnet_channels) - s - 1], 
                        use_res=True, 
                        norm_fn=norm_fn
                    )
                )

            enc_channels = channels[s] + ca_out_channels[s]
            point_enc_channels = pointnet_channels[s]
            dec_channels = channels[len(channels) - s - 1]
            point_dec_channels = pointnet_channels[len(pointnet_channels) - s - 1]

        final_in_channels = (
            channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        )
        final_in_point_channels = (
            pointnet_channels[-1] if not self.cls_mode else pointnet_channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.point_final = (
            nn.Sequential(
                nn.Linear(final_in_point_channels, final_in_point_channels),
                nn.BatchNorm1d(final_in_point_channels),
                nn.ReLU(inplace=True),
                nn.Linear(final_in_point_channels, num_classes),
            )
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            if m.affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, PDBatchNorm):
            if self.zero_init:
                nn.init.constant_(m.modulation[-1].weight, 0)
                nn.init.constant_(m.modulation[-1].bias, 0)
   
    def voxel_min_coord(self, input_dict):
        offset = input_dict["offset"]
        min_coords = torch.zeros(offset.shape[0], 3).to(offset.device)
        voxel_offset = torch.cat((torch.tensor([0]).to(offset.device), offset), dim=0)
        for i in range(voxel_offset.shape[0] - 1):
            min_coords[i] = torch.min(input_dict["coord"][voxel_offset[i]:voxel_offset[i+1]], dim=0)[0]

        return min_coords

    def forward(self, input_dict, voxel_index=None):
        # num_gpu = faiss.get_num_gpus()
        # print("num_gpu: ", num_gpu)
        if voxel_index == None:
            # voxel_quantizer = faiss.IndexFlatL2(3)
            # voxel_index = faiss.IndexIVFFlat(voxel_quantizer, 3, 10, faiss.METRIC_L2)
            # res = faiss.StandardGpuResources()
            # voxel_index = faiss.index_cpu_to_gpu(res, 0, voxel_index)

            # ngpu = 2
            # resources = [faiss.StandardGpuResources() for i in range(ngpu)]
            # voxel_index = faiss.index_cpu_to_gpu_multiple_py(resources, voxel_index)
            # faiss.contrib.torch_utils.handle_torch_Index(voxel_index)

            voxel_quantizer = faiss.IndexFlatL2(3)
            voxel_index = faiss.IndexIVFFlat(voxel_quantizer, 3, 10, faiss.METRIC_L2)
            voxel_index = faiss.index_cpu_to_gpus_list(voxel_index, gpus=[point_coord.device.index])
            faiss.contrib.torch_utils.handle_torch_Index(voxel_index)

            
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        condition = input_dict["condition"][0]
        context = input_dict["context"] if "context" in input_dict.keys() else None
        point_coord = input_dict["point_patch_coord"]
        point_feat = input_dict["point_patch_feat"]
        point_offset = input_dict["point_patch_offset"]
        sensor = input_dict["sensor"][0]
        min_coords = self.voxel_min_coord(input_dict)

        voxel_gpu_index = voxel_index

        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.conv_input[self.sensors.index(sensor)]([x, condition, context])
        skips = [x]

        f = self.point_input[self.sensors.index(sensor)](point_feat, condition)
        pfo = [point_coord, f, point_offset]
        pfo_skips = [pfo]

        for s in range(self.num_stages):
            if "point_patch_label" in input_dict.keys():
                pfo = self.hypernet[s](x, pfo, condition, min_coords, self.grid_size*2**s, voxel_gpu_index)
                pfo = self.point_down[s](pfo, condition)
                pfo[1] = self.point_enc[s](pfo[1], condition)
                pfo_skips.append(pfo)
                _, f, _ = pfo
                feature = f[:self.geo_pool_max_size[s]//2] if f.shape[0] > self.geo_pool_max_size[s]//2 else f

                if self.geo_pool[self.sensors.index(sensor)][s] == None:
                    self.geo_pool[self.sensors.index(sensor)][s] = feature.clone().detach().float()
                else:
                    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
                    geo_pool = self.geo_pool[self.sensors.index(sensor)][s].clone().detach() # (n, c)

                    cos = cosine_similarity(geo_pool.unsqueeze(-1), feature.clone().detach().transpose(0,1).unsqueeze(0)) # (n, m)
                    cos_max, cos_idx = torch.max(cos, dim=0) # (m), (m)
                    new_feature = feature[cos_max < self.thresold].clone().detach() # (m', c)
                    update_feature = feature[cos_max >= self.thresold].clone().detach() # (m-m', c)
                    update_idx = cos_idx[cos_max >= self.thresold] # (m-m')
                    geo_pool[update_idx] = geo_pool[update_idx] + self.update_rate * update_feature
                    
                    if geo_pool.shape[0] + new_feature.shape[0] <= self.geo_pool_max_size[s]:
                        geo_pool = torch.cat((geo_pool, new_feature), dim=0)
                    elif geo_pool.shape[0] < self.geo_pool_max_size[s]:
                        select_new_feature = new_feature[:self.geo_pool_max_size[s]-geo_pool.shape[0]]
                        geo_pool = torch.cat((geo_pool, select_new_feature), dim=0)
                        new_cos = cosine_similarity(geo_pool.unsqueeze(-1), new_feature[self.geo_pool_max_size[s]-geo_pool.shape[0]:].transpose(0,1).unsqueeze(0)) # (n', m'')
                        new_cos_idx = torch.max(new_cos, dim=0)[1]
                        geo_pool[new_cos_idx] = geo_pool[new_cos_idx] + self.update_rate * new_feature[self.geo_pool_max_size[s] - geo_pool.shape[0]:]
                    else:
                        new_cos = cosine_similarity(geo_pool.unsqueeze(-1), new_feature.transpose(0,1).unsqueeze(0)) # (n', m)
                        new_cos_idx = torch.max(new_cos, dim=0)[1] # (m)
                        geo_pool[new_cos_idx] = geo_pool[new_cos_idx] + self.update_rate * new_feature

                    self.geo_pool[self.sensors.index(sensor)][s] = geo_pool.detach().float()    

            x = self.down[s]([x, condition, context])
            geo_pool = self.geo_pool[self.sensors.index(sensor)][s].clone().detach().float()
            x = x.replace_feature(self.ca[s](geo_pool, x.features))
            x, _, _ = self.enc[s]([x, condition, context])
            skips.append(x)

        x = skips.pop(-1)
        if "point_patch_label" in input_dict.keys():
            pfo_previous = pfo_skips.pop(-1)
            pfo_previous[1] = self.point_dec_input(pfo_previous[1], condition)
        
        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s]([x, condition, context])
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x, _, _ = self.dec[s]([x, condition, context])
                if "point_patch_label" in input_dict.keys():
                    pfo = pfo_skips.pop(-1)
                    f = self.point_up[s](pfo, pfo_previous, condition)
                    f = self.point_dec[s](f, condition)
                    pfo_previous = [pfo[0], f, pfo[2]]

        x = self.final(x)
        if "point_patch_label" in input_dict.keys():
            f = self.point_final(f)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )

        if "point_patch_label" in input_dict.keys():
            return x.features, f
        else:
            return x.features

