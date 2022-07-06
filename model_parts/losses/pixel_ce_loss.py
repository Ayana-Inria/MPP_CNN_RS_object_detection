import torch
from torch import nn
from torch.nn.functional import cross_entropy

from model_parts.losses.focalloss import focal_loss


class PixelCELoss(nn.Module):
    def __init__(self, focal_loss=False, focal_loss_args=None):
        """

        :param mask_sigma: value for simga to apply around objects, if auto will take param 0 as size and use sigma=size/4
        :param mask_cutoff_dist: maximum distance for sigma computation, if sigma is auto then cutoff is chosen where
        the gaussian < 1e-3
        """
        super(PixelCELoss, self).__init__()
        self.focal_loss = focal_loss or False
        self.focal_loss_args = focal_loss_args

    def forward(self, inputs, targets, loss_mask, return_maps=False):
        """
        :param inputs: prediction tensor Tensor of shape (B,C,H,W), where C is the histogram size
        for each parameter
        :param targets: integer class for each feature Tensor of shape (B,H,W)
        :param return_maps: if true the returned dict has key 'loss_mask' and 'pixel_loss'
        :return: Loss dictionary

        """

        return_dict = {}

        if type(inputs) is list and type(targets) is list:
            loss = 0
            for i, (f_inputs, f_targets) in enumerate(zip(inputs, targets)):
                if self.focal_loss:
                    feat_pp_loss = focal_loss(f_inputs, f_targets, reduction='none', **self.focal_loss_args)
                else:
                    feat_pp_loss = cross_entropy(f_inputs, f_targets, reduction='none')
                feat_pp_loss_masked = feat_pp_loss * loss_mask
                feat_loss = torch.sum(feat_pp_loss_masked, dim=(1, 2))
                feat_loss = torch.mean(feat_loss)
                loss += feat_loss

                return_dict[f'loss_feat{i}'] = feat_loss
                if return_maps:
                    return_dict[f'pixel_loss_feat{i}'] = feat_pp_loss_masked

        else:
            pp_loss = cross_entropy(inputs, targets, reduction='none')
            pp_loss_masked = pp_loss * loss_mask
            loss = torch.mean(torch.sum(pp_loss_masked, dim=(1, 2)))

            if return_maps:
                return_dict[f'pixel_loss'] = pp_loss_masked

        return_dict['loss'] = loss
        return return_dict
