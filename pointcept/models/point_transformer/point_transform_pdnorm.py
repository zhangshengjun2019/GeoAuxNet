"""
Point Transformer V1 for Semantic Segmentation

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import einops
import pointops

from ..builder import MODELS
from .utils import LayerNorm1d

class PDBatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-3,
        momentum=0.01,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        affine=True,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
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

    def forward(self, feat, condition=None):
        if self.decouple:
            assert condition in self.conditions
            bn = self.bns[self.conditions.index(condition)]
        else:
            bn = self.bn
        feat = bn(feat)
        return feat

class PDLayerNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-3,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        affine=True,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.affine = affine
        if self.decouple:
            self.bns = nn.ModuleList(
                [
                    nn.LayerNorm(
                        num_features,
                        eps=eps,
                        elementwise_affine=affine,
                    )
                    for _ in conditions
                ]
            )
        else:
            self.bn = nn.LayerNorm(
                        num_features=num_features,
                        eps=eps,
                        elementwise_affine=affine,
                    )

    def forward(self, feat, condition=None):
        if self.decouple:
            assert condition in self.conditions
            bn = self.bns[self.conditions.index(condition)]
        else:
            bn = self.bn
        feat = bn(feat)
        return feat   

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, ln=None):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        # self.linear_p
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            ln(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )

        self.linear_w = nn.Sequential(
            ln(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            ln(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo, condition) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(
            x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True
        )
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        # p_r = self.linear_p(p_r)
        for i in range(len(self.linear_p)):
            if i == 1:
                p_r = self.linear_p[i](p_r, condition)
            else:
                p_r = self.linear_p[i](p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        # w = self.linear_w(r_qk)  # (n, nsample, c)
        for i in range(len(self.linear_w)):
            if i == 0 or i == 3:
                r_qk = self.linear_w[i](r_qk, condition)
            else:
                r_qk = self.linear_w[i](r_qk)
        w = self.softmax(r_qk)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16, bn=None):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = bn(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo, condition):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
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
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None, bn=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                bn(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                bn(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                bn(out_planes),
                nn.ReLU(inplace=True),
            )

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
            for i in range(len(self.linear1)):
                if i == 1:
                    x = self.linear1[i](x, condition)
                else:
                    x = self.linear1(x)
        else:
            
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            assert p1.shape[0] == x1.shape[0]
            for i in range(len(self.linear1)):
                if i == 1:
                    x1 = self.linear1[i](x1, condition)
                else:
                    x1 = self.linear1(x1)
            for i in range(len(self.linear2)):
                if i == 1:
                    x2 = self.linear2[i](x2, condition)
                else:
                    x2 = self.linear2[i](x2)
            x = x1 + pointops.interpolation(
                p2, p1, x2, o2, o1
            )
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, bn=None, ln=None):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = bn(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample, ln)
        self.bn2 = bn(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = bn(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo, condition):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x), condition))
        x = self.relu(self.bn2(self.transformer([p, x, o], condition), condition))
        x = self.bn3(self.linear3(x), condition)
        x += identity
        x = self.relu(x)
        return [p, x, o]


