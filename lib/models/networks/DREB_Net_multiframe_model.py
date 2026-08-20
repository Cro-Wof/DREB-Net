"""Three-frame DREB with feature-level temporal alignment and fusion."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.ops import DeformConv2d

from .DREB_Net_model import DREB_Net


class TemporalDeformAlign(nn.Module):
    """Align one neighboring S2 feature map to the center-frame S2 map."""

    def __init__(self, channels=128, kernel_size=3):
        super(TemporalDeformAlign, self).__init__()
        self.kernel_size = kernel_size
        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size
        self.offset_mask = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                offset_channels + mask_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, neighbor, center):
        offset_mask = self.offset_mask(torch.cat([neighbor, center], dim=1))
        offset_channels = 2 * self.kernel_size * self.kernel_size
        offset = offset_mask[:, :offset_channels]
        mask = offset_mask[:, offset_channels:].sigmoid()
        aligned = self.deform(neighbor, offset, mask)
        return self.refine(aligned)


class TemporalGateFusion(nn.Module):
    """Center-guided, residual temporal fusion for three aligned features."""

    def __init__(self, channels=128, embed_channels=32):
        super(TemporalGateFusion, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(channels, embed_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, 1, kernel_size=1),
        )
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        # Start as the center-frame path.  This makes a newly added temporal
        # branch safe to initialize beside a pretrained single-frame DREB.
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, center, previous, following):
        features = [previous, center, following]
        center_embed = self.embed(center)
        logits = []
        for feature in features:
            logits.append(self.gate(torch.cat([center_embed, self.embed(feature)], dim=1)))
        weights = torch.softmax(torch.cat(logits, dim=1), dim=1)
        weighted = [feature * weights[:, index:index + 1] for index, feature in enumerate(features)]
        return center + self.fuse(torch.cat(weighted, dim=1))


class DREB_Net_MF(DREB_Net):
    """DREB with a three-frame feature-alignment branch.

    The original DREB_Net class remains unchanged.  Neighboring frames only
    use the shallow/deblur encoder; the center frame uses the original
    detection backbone and center-frame skip connections.
    """

    def __init__(self, num_blocks=None, width_multiplier=None, override_groups_map=None,
                 deploy=False, use_checkpoint=False, heads=None, head_conv=None):
        if num_blocks is None:
            num_blocks = [4, 6, 16, 1]
        if width_multiplier is None:
            width_multiplier = [1, 1, 1, 1]
        super(DREB_Net_MF, self).__init__(
            num_blocks=num_blocks,
            width_multiplier=width_multiplier,
            override_groups_map=override_groups_map,
            deploy=deploy,
            use_checkpoint=use_checkpoint,
            heads=heads,
            head_conv=head_conv,
        )
        self.temporal_align = TemporalDeformAlign(channels=128)
        self.temporal_fusion = TemporalGateFusion(channels=128)

    def _run_stage(self, blocks, value):
        for block in blocks:
            if self.use_checkpoint and value.requires_grad:
                value = checkpoint.checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        return value

    def forward(self, x, mode):
        if x.ndim != 5:
            raise ValueError('DREB_Net_MF expects [B, T, C, H, W], got {}'.format(tuple(x.shape)))
        if x.size(1) != 3 or x.size(2) != 3:
            raise ValueError('DREB_Net_MF expects exactly three RGB frames')

        batch_size, num_frames, channels, height, width = x.shape
        shallow = x.reshape(batch_size * num_frames, channels, height, width)
        shallow = self.stage0(shallow)
        s0_all = shallow
        s1_all = self.deblur_down1(s0_all)
        s2_all = self.deblur_down2(s1_all)

        s0_all = s0_all.reshape(batch_size, num_frames, s0_all.size(1), s0_all.size(2), s0_all.size(3))
        s1_all = s1_all.reshape(batch_size, num_frames, s1_all.size(1), s1_all.size(2), s1_all.size(3))
        s2_all = s2_all.reshape(batch_size, num_frames, s2_all.size(1), s2_all.size(2), s2_all.size(3))

        s0 = s0_all[:, 1]
        s1 = s1_all[:, 1]
        s2 = s2_all[:, 1]
        previous = self.temporal_align(s2_all[:, 0], s2)
        following = self.temporal_align(s2_all[:, 2], s2)
        s2_temporal = self.temporal_fusion(s2, previous, following)

        out = self._run_stage(self.stage1, s0)
        out = self._run_stage(self.stage2, out)
        out_LFAMM = self.LFAMM(out)
        out = self.MAGFF_attention(out, s2_temporal)
        out = out_LFAMM + out
        out = self._run_stage(self.stage3, out)
        out = self._run_stage(self.stage4, out)

        out = self.deconv_layers1(out)
        out = self.deconv_layers2(out)
        out = self.deconv_layers3(out)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(out)

        if mode == 'val':
            return [ret]
        if mode == 'train':
            down3 = self.deblur_down3(s2_temporal)
            down4 = self.deblur_down4(down3)
            up1 = self.deblur_up1(down4, down3)
            up2 = self.deblur_up2(up1, s2_temporal)
            up3 = self.deblur_up3(up2, s1)
            up4 = self.deblur_up4(up3, s0)
            deblur_out = self.deblur_up5(up4, None)
            return [ret], deblur_out
        raise ValueError('mode not eq train/val!!!')


def create_DREB_Net_multiframe_detect(deploy=False, use_checkpoint=False, heads=None, head_conv=None):
    print('create_DREB_Net_MF')
    return DREB_Net_MF(
        override_groups_map=None,
        deploy=deploy,
        use_checkpoint=use_checkpoint,
        heads=heads,
        head_conv=head_conv,
    )
