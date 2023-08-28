import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.spatial_color_alignment as sca_utils
from utils.spatial_color_alignment import get_gaussian_kernel, match_colors
import lpips
# from utils.ssim import cal_ssim
from utils.data_format_utils import numpy_to_torch, torch_to_numpy
import numpy as np
import lpips
from utils.warp import warp
import utils.mssim as msssim 



class L2(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.mse_loss(pred_m, gt_m)
        else:
            mse = F.mse_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse + 1e-6

"""
class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, valid=None):
        assert pred.dim() == 4 and pred.shape == gt.shape
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr
"""
class PixelWiseError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
        calculation """
    def __init__(self, metric='l1', boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        elif metric == 'l2_sqrt':
            def l2_sqrt(pred, gt):
                return (((pred - gt) ** 2).sum(dim=-3)).sqrt().mean()
            self.loss_fn = l2_sqrt
        elif metric == 'charbonnier':
            def charbonnier(pred, gt):
                eps = 1e-3
                return ((pred - gt) ** 2 + eps**2).sqrt().mean()
            self.loss_fn = charbonnier
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Valid indicates image regions which should be used for loss calculation
        if valid is None:
            err = self.loss_fn(pred, gt)
        else:
            err = self.loss_fn(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err
class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = PixelWiseError(metric='l2', boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        if getattr(self, 'max_value', 1.0) is not None:
            psnr = 20 * math.log10(getattr(self, 'max_value', 1.0)) - 10.0 * mse.log10()
        else:
            psnr = 20 * gt.max().log10() - 10.0 * mse.log10()

        if torch.isinf(psnr) or torch.isnan(psnr):
            print('invalid psnr')

        return psnr

    def forward(self, pred, gt, valid=None):
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]

        psnr_all = [p for p in psnr_all if not (torch.isinf(p) or torch.isnan(p))]

        if len(psnr_all) == 0:
            psnr = 0
        else:
            psnr = sum(psnr_all) / len(psnr_all)
        return psnr

class SSIM(nn.Module):
    def __init__(self, boundary_ignore=None, use_for_loss=True):
        super().__init__()
        self.ssim = msssim.SSIM(spatial_out=True)
        self.boundary_ignore = boundary_ignore
        self.use_for_loss = use_for_loss

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        loss = self.ssim(pred, gt)

        if valid is not None:
            valid = valid[..., 5:-5, 5:-5]  # assume window size 11

            eps = 1e-12
            elem_ratio = loss.numel() / valid.numel()
            loss = (loss * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)
        else:
            loss = loss.mean()

        if self.use_for_loss:
            loss = 1.0 - loss
        return loss


class LPIPS(nn.Module):
    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.bgr2rgb = bgr2rgb

        if type == 'alex':
            self.loss = lpips.LPIPS(net='alex')
        elif type == 'vgg':
            self.loss = lpips.LPIPS(net='vgg')
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.bgr2rgb:
            pred = pred[..., [2, 1, 0], :, :].contiguous()
            gt = gt[..., [2, 1, 0], :, :].contiguous()

        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        loss = self.loss(pred, gt)

        return loss.mean()

#################################################################################
# Compute aligned L1 loss
#################################################################################

class AlignedL1(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        l1 = F.l1_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        l1 = l1 + eps
        elem_ratio = l1.numel() / valid.numel()
        l1 = (l1 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return l1


class AlignedL1_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l1 = AlignedL1(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        
    def forward(self, pred, gt, burst_input):
        L1_all = [self.l1(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        L1_loss = sum(L1_all) / len(L1_all)
        return L1_loss

# torch.Size([1, 14, 4, 80, 80]) torch.Size([1, 3, 640, 640]) torch.Size([1, 3, 640, 640])
def make_patches(output, labels, burst, patch_size=48): 
    num_frames = burst.size(1)   
    stride = patch_size-(burst.size(-1)%patch_size) # 16   
    # [14, 4, 48, 48]
    burst1 = burst[0].unfold(2,patch_size,stride).unfold(3,patch_size,stride).contiguous()  # torch.Size([14, 4, 3, 3, 48, 48])         
    burst1 = burst1.view(num_frames,4,burst1.size(2)*burst1.size(3),patch_size,patch_size).permute(2,0,1,3,4)            
    output1 = output.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    output1 = output1.view(3,output1.size(2)*output1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)
    labels1 = labels.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    labels1 = labels1[0].view(3,labels1.size(2)*labels1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)
    return output1, labels1, burst1

#################################################################################
# Compute aligned PSNR, LPIPS, and SSIM
#################################################################################


class AlignedPred(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        return pred_warped_m, gt, valid


class AlignedL2(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor
        # flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear') * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)
        # frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear')

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # print(pred_warped_m[0, 0, 0, 0:10])

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        # Valid indicates image regions which should be used for loss calculation
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')
        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse

class AlignedL2_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        
    def forward(self, pred, gt, burst_input):
        L2_all = [self.l2(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        L2_loss = sum(L2_all) / len(L2_all)
        return L2_loss
        
class AlignedSSIM_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        

    def ssim(self, pred, gt, burst_input):
        
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)

        gt = gt[0, 0, :, :]
        pred_warped_m = pred_warped_m[0, 0, :, :]

        mssim,ssim_map = cal_ssim(pred_warped_m*255, gt*255)
        ssim_map = torch.from_numpy(ssim_map).float()
        valid = torch.squeeze(valid)

        eps = 1e-12
        elem_ratio = ssim_map.numel() / valid.numel()
        ssim = (ssim_map * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return 1 - ssim

    def forward(self, pred, gt, burst_input):
        ssim_all = [self.ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        ssim = sum(ssim_all) / len(ssim_all)
        return ssim

class AlignedLPIPS_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    def lpips(self, pred, gt, burst_input):

        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_vgg(var1, var2)
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips

class AlignedPSNR(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value    

    def psnr(self, pred, gt, burst_input):
        
        #### PSNR
        mse = self.l2(pred, gt, burst_input) + 1e-12
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, burst_input):
        # torch.Size([9, 3, 384, 384]) torch.Size([9, 3, 384, 384]) torch.Size([9, 14, 4, 48, 48])
        pred, gt, burst_input = make_patches(pred, gt, burst_input)
        psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr

import cv2
import numpy as np
from scipy import signal

def cal_ssim(img1, img2):
    
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'same')
    mu2 = signal.convolve2d(img2, window, 'same')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'same') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'same') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'same') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim,ssim_map
"""
# Assuming single channel images are read. For RGB image, uncomment the following commented lines
img1 = cv2.imread('location_noisy',0)
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('location_clean',0)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
"""


class AlignedSSIM(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        

    def ssim(self, pred, gt, burst_input):
        
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)

        gt = gt[0, 0, :, :].cpu().numpy()
        pred_warped_m = pred_warped_m[0, 0, :, :].cpu().numpy()

        mssim,ssim_map = cal_ssim(pred_warped_m*255, gt*255)
        ssim_map = torch.from_numpy(ssim_map).float()
        valid = torch.squeeze(valid.cpu())

        eps = 1e-12
        elem_ratio = ssim_map.numel() / valid.numel()
        ssim = (ssim_map * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return ssim

    def forward(self, pred, gt, burst_input):
        ssim_all = [self.ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        ssim = sum(ssim_all) / len(ssim_all)
        return ssim

class AlignedLPIPS(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def lpips(self, pred, gt, burst_input):
        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_alex(var1.cpu(), var2.cpu())
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips






