import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

from model_parts.losses.focalloss import binary_focal_loss_with_logits

eps = 1e-5


class PosCELoss(nn.Module):
    def __init__(self, n_classes):
        super(PosCELoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target, return_maps=False, ):
        """
        :param input: prediction tensor Tensor of shape (B,C,H,W), where C is the histogram size
        for each parameter
        :param distance_map: distance for each pixel, Tensor of shape (B,H,W)
        :param return_maps: if true the returned dict has key 'loss_mask' and 'pixel_loss'
        :return: Loss dictionary

        """
        one_hot = torch.nn.functional.one_hot(torch.squeeze(target), num_classes=self.n_classes)  # -> B,H,W,n_classes
        weights = 1 - torch.mean(one_hot.type(torch.float32), dim=(0, 1, 2))  # -> n_classes
        loss = nn.functional.cross_entropy(input, target, weight=weights)

        return_dict = {
            'losses': loss
        }

        return return_dict


class PointingVectorLoss(nn.Module):
    def __init__(self, learn_mask: bool, compute_mask: bool, balanced_mask_loss=False, focal_loss=False,
                 vec_loss_on_prod=False):
        super(PointingVectorLoss, self).__init__()
        self.learn_mask = learn_mask
        self.compute_mask = compute_mask
        self.balanced_mask_loss = balanced_mask_loss or False
        self.focal_loss = focal_loss or False
        self.vec_loss_on_prod = vec_loss_on_prod or False
        if self.vec_loss_on_prod:
            assert self.compute_mask

    def forward(self, output, target_vec, target_mask=None, div_score=None, center_bin_map=None):
        """
        :param output: prediction tensor Tensor of shape (B,C,H,W), where C is the histogram size
        for each parameter
        :param target_vec: target vectors pointing towards the center of each object

        :return: Loss dictionary

        """

        output_mask = output[:, 2, :, :]
        output_vec = output[:, :2, :, :]  # input is B,C,H,W, take first two channels

        if self.vec_loss_on_prod:
            output_mask_sig = torch.sigmoid(output_mask)
            prod = output_vec * torch.stack([output_mask_sig, output_mask_sig], dim=1)
            pixel_loss = torch.square(prod - target_vec)
            vec_loss = torch.mean(pixel_loss)
        else:
            pixel_loss = torch.square(output_vec - target_vec)
            if self.compute_mask:
                pixel_loss = torch.mean(pixel_loss, dim=1) * target_mask
            vec_loss = torch.mean(pixel_loss)

        return_dict = {
            'vec_loss': vec_loss,
            'loss': vec_loss
        }

        if self.learn_mask:

            if self.focal_loss:

                mask_loss = binary_focal_loss_with_logits(output_mask, target_mask, reduction='mean')
            else:
                output_mask_sig = torch.sigmoid(output_mask)
                if not self.balanced_mask_loss:
                    mask_loss = binary_cross_entropy(output_mask_sig, target_mask)
                else:
                    beta = 1 - torch.sum(target_mask) / torch.numel(target_mask)
                    mask_loss = -beta * target_mask * torch.log(output_mask_sig + eps) \
                                - (1 - beta) * (1 - target_mask) * torch.log(1 - output_mask_sig + eps)
                    # pt = output_mask * target_mask + (1-target_mask) * (1-output_mask)
                    # beta = 0.999
                    # ny = torch.sum(target_mask)
                    # mask_loss = - ((1-beta) / (1-torch.pow(beta,ny))) * torch.log(pt)
                    mask_loss = torch.mean(mask_loss)
            return_dict['mask_loss'] = mask_loss
            return_dict['loss'] = return_dict['loss'] + mask_loss

        if div_score is not None:
            assert center_bin_map is not None
            div_score = torch.squeeze(div_score, dim=1)  # to (B,H,W)
            if self.focal_loss:
                div_loss = binary_focal_loss_with_logits(div_score, center_bin_map, reduction='mean')
            else:
                div_score_sig = torch.sigmoid(div_score)
                if not self.balanced_mask_loss:
                    div_loss = binary_cross_entropy(div_score_sig, center_bin_map)
                else:
                    beta = 1 - torch.sum(center_bin_map) / torch.numel(center_bin_map)
                    div_loss = -beta * center_bin_map * torch.log(div_score_sig + eps) \
                               - (1 - beta) * (1 - center_bin_map) * torch.log(1 - div_score_sig + eps)
                    div_loss = torch.mean(div_loss)

            return_dict['div_loss'] = div_loss
            return_dict['loss'] = return_dict['loss'] + div_loss

        return return_dict
