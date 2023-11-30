import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock, Feature, Convnext_Encoder_stereo, Convnext_Encoder_stereo_cnet
from core.corr import CorrBlock1D_Cost_Volume
from core.utils.utils import coords_grid, upflow8, updisp4
from core.submodule import *

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class MCStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.max_disp = 192
        self.args.K_value = 3
        context_dims = args.hidden_dims
        if args.feature_extractor == 'resnet':
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        elif args.feature_extractor == 'convnext':
            self.fnet = Convnext_Encoder_stereo()
            self.cnet = Convnext_Encoder_stereo_cnet()
        self.update_block_0 = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, cor_dim=12,
                                                    disp_dim=self.args.K_value, K_value=self.args.K_value)
        self.update_block_1 = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, cor_dim=4,
                                                    disp_dim=self.args.K_value, K_value=self.args.K_value)
        self.update_block_2 = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, cor_dim=2, disp_dim=1,
                                                    K_value=1)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])
        self.uniform_sampler = UniformSampler()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        up_disp = F.unfold(factor * disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)
        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.feature_extractor == 'resnet':
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
            elif self.args.feature_extractor == 'convnext':
                cnet_list = self.cnet(image1)
                fmap1 = self.fnet(image1)
                fmap2 = self.fnet(image2)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]

        corr_block = CorrBlock1D_Cost_Volume
        fmap1, fmap2 = fmap1.float(), fmap2.float()

        init_corr_volume = build_correlation_volume(fmap1, fmap2, self.args.max_disp // 4)
        corr_fn = corr_block(init_corr_volume, num_levels=2)
        cost_volume = init_corr_volume.squeeze(1)
        prob = F.softmax(cost_volume, dim=1)
        # disp_init = disparity_regression(prob, self.args.max_disp // 4)
        value, disp_topk = torch.topk(prob, dim=1, k=self.args.K_value)

        disp_predictions = []
        context = net_list[0]
        for itr in range(iters):
            disp_topk = disp_topk.detach().float()
            local_cost_list = []
            corr_list = []
            # Search Radius
            if itr < 6:
                rt = 12
            elif itr < 16:
                rt = 4
            else:
                rt = 2
            # Multi-peak Lookup
            for k in range(disp_topk.shape[1]):
                disp = disp_topk[:, k, :, :].unsqueeze(dim=1)
                corr_pyramid = corr_fn(disp, rt)
                local_cost_list.append(corr_pyramid[0])
                corr = torch.cat(corr_pyramid, dim=1)
                corr_list.append(corr)
            local_cost = torch.cat(local_cost_list, dim=1)
            corr = torch.cat(corr_list, dim=1)
            with autocast(enabled=self.args.mixed_precision):
                if itr < 6:
                    net_list, up_mask, delta_local_cost = self.update_block_0(net_list, inp_list, corr, disp_topk,
                                                                              context,
                                                                              iter32=self.args.n_gru_layers == 3,
                                                                              iter16=self.args.n_gru_layers >= 2)
                elif itr < 16:
                    net_list, up_mask, delta_local_cost = self.update_block_1(net_list, inp_list, corr, disp_topk,
                                                                              context,
                                                                              iter32=self.args.n_gru_layers == 3,
                                                                              iter16=self.args.n_gru_layers >= 2)
                else:
                    net_list, up_mask, delta_local_cost = self.update_block_2(net_list, inp_list, corr, disp_topk,
                                                                              context,
                                                                              iter32=self.args.n_gru_layers == 3,
                                                                              iter16=self.args.n_gru_layers >= 2)
                k_value = 1 if itr >= 15 else self.args.K_value
            local_cost = local_cost + delta_local_cost
            prob = F.softmax(local_cost, dim=1)
            disparity_samples_list = []
            for dk in range(disp_topk.shape[1]):
                disp_t = disp_topk[:, dk, :, :].unsqueeze(dim=1)
                min_disparity = disp_t - rt
                max_disparity = disp_t + rt
                d_samples = self.uniform_sampler(min_disparity, max_disparity, 2 * rt + 1)
                disparity_samples_list.append(d_samples)
            disparity_samples = torch.cat(disparity_samples_list, dim=1)
            disp = torch.sum(prob * disparity_samples, dim=1, keepdim=True)
            _, disp_topk_index = torch.topk(prob, dim=1, k=k_value)
            disp_topk = torch.gather(disparity_samples, 1, disp_topk_index)
            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue
            # upsample predictions
            if up_mask is None:
                disp_up = updisp4(disp)
            else:
                disp_up = self.upsample_disp(disp, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp, disp_predictions[-1]
        return disp_predictions


