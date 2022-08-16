import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Statistician(nn.Module):
    def __init__(self, code=['MAX', 'OM1','OM2','CM2', 'CM3']):
        super(Statistician, self).__init__()
        self.code  = code

    def forward(self, x):
        tmp = []
        om1 = torch.mean(x, dim=2, keepdim=True)
        for i, c in enumerate(self.code):
            if c == 'MAX':
                tmp.append(torch.max(x, dim=2, keepdim=True)[0])
            elif c[:2] == 'OM': #origin moment
                order = int(c[2:])
                tmp.append(torch.mean(torch.pow(x, order), 2, keepdim=True))
            elif c[:2] == 'CM': #central moment
                order = int(c[2:])
                tmp.append(torch.mean(torch.pow(x-om1, order), 2, keepdim=True))
        # last_om = x
        # last_om_order = 1
        # x1 = x - om1
        # last_cm = x1
        # last_cm_order = 1
        # for i, c in enumerate(self.code):
        #     if c == 'MAX':
        #         tmp.append(torch.max(x, dim=2, keepdim=True)[0])
        #     elif c[:2] == 'OM': #origin moment
        #         order = int(c[2:])
        #         for _ in range(order - last_om_order):
        #             last_om = x * last_om
        #         tmp.append(torch.mean(last_om, dim=2, keepdim=True))
        #         last_om_order = order
        #     elif c[:2] == 'CM': #central moment
        #         order = int(c[2:])
        #         for _ in range(order - last_cm_order):
        #             last_cm = x1 * last_cm
        #         tmp.append(torch.mean(last_cm, dim=2, keepdim=True))
        #         last_cm_order = order
        return torch.cat(tmp, dim=2)


class Aggregator(nn.Module):
    def __init__(self, type=0):
        super(Aggregator, self).__init__()
        self.type = type

    def forward(self, x):
        if self.type == 0:
            return torch.sum(x, dim=2, keepdim=True)


class StatisAttention(nn.Module):
    def __init__(self, nc=1024, ns=5, dim=3):
        #for pointnet dim=3 for pointnet++ dim=4
        super(StatisAttention, self).__init__()
        self.dim = dim
        self.nc = nc #number of feature channels
        self.ns = ns #number of statistics
        self.conv1 = torch.nn.Conv1d(nc, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 1, 1)
        self.conv3 = torch.nn.Conv1d(ns, ns, 1)
        #self.fc1 = nn.Linear(ns, ns)

    def forward(self, x):
        y = x
        if self.dim == 3:
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(-1, self.ns, 1)
            x = self.conv3(x)
            x = F.sigmoid(x)
            x = x.view(-1, 1, self.ns)
        if self.dim == 4:
            s = x.shape
            x = x.view(s[0], s[1], -1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(s[0], s[2], s[3])
            x = self.conv3(x)
            x = F.sigmoid(x)
            x = x.view(s[0], 1, s[2], s[3])
        return y*x


class Hosa(nn.Module):
    def __init__(self, np=1024, nc=1024, dim=3, code=['MAX', 'OM1', 'CM2', 'CM3', 'CM4'], atype=0, use_sa=False):
        super(Hosa, self).__init__()
        self.sta_code = code
        self.statistician = Statistician(code=self.sta_code)
        self.ns = len(code)
        self.np = np
        self.use_sa = use_sa
        if use_sa:
            self.statis_attention = StatisAttention(nc=nc, ns=self.ns, dim=dim)
        self.atype = atype
        self.aggregator = Aggregator(type=self.atype)

    def forward(self, x):
        x = self.statistician(x)
        if self.use_sa:
            x = self.statis_attention(x)
        x = self.aggregator(x)
        return x