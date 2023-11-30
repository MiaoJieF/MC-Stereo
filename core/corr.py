import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class CorrBlock1D_Cost_Volume:
    def __init__(self, init_corr, num_levels=2):
        self.num_levels = num_levels
        self.init_corr_pyramid = []
        b, c, d, h, w = init_corr.shape
        init_corr = init_corr.permute(0, 3, 4, 1, 2).reshape(b * h * w, 1, 1, d)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels):
            init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, rt):
        b, _, h, w = disp.shape
        point_num = 2 * rt + 1
        out_pyramid = []
        for i in range(self.num_levels):
            init_corr = self.init_corr_pyramid[i]
            dx = torch.linspace(-rt, rt, point_num)
            dx = dx.view(1, 1, point_num, 1).to(disp.device)
            x0 = dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0, y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, disp_lvl)
            init_corr = init_corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous().float()
            out_pyramid.append(init_corr)
        return out_pyramid
