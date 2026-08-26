"""Three-frame DREB with a lightweight EDVR-style temporal branch.

The DREB detection backbone remains center-frame-only. Neighboring frames
share the existing shallow/deblur encoder, are aligned to the center frame
with a two-level PCD module (s2 -> s1), and are fused with center-guided
temporal-spatial attention.
"""

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchvision.ops import DeformConv2d

from .DREB_Net_model import DREB_Net


class TemporalDeformAlign(nn.Module):
    """Two-level PCD alignment for one neighbor and the center frame.

    The first implementation keeps the EDVR coarse-to-fine idea small enough
    for a 24GB single-GPU setting:

    * L3: align s2 (1/8 resolution, 128 channels);
    * L2: align s1 (1/4 resolution, 64 channels), guided by upsampled L3;
    * cascade: refine the final s1 alignment against the center feature.

    Parameters are shared by the previous and following frames.
    """

    def __init__(self, s2_channels=128, s1_channels=64, kernel_size=3):
        super(TemporalDeformAlign, self).__init__()
        self.kernel_size = kernel_size
        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size
        output_channels = offset_channels + mask_channels

        # L3 / coarse alignment.
        self.offset_mask_l3 = nn.Sequential(
            nn.Conv2d(s2_channels * 2, s2_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s2_channels, output_channels, kernel_size=3, padding=1),
        )
        self.deform_l3 = DeformConv2d(
            s2_channels,
            s2_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.refine_l3 = nn.Sequential(
            nn.Conv2d(s2_channels, s2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(s2_channels),
            nn.ReLU(inplace=True),
        )

        # Project coarse features before using them in the s1 predictor.
        self.coarse_to_s1 = nn.Conv2d(s2_channels, s1_channels, kernel_size=1)

        # L2 / fine alignment.
        l2_input_channels = s1_channels * 3
        self.offset_mask_l2 = nn.Sequential(
            nn.Conv2d(l2_input_channels, s1_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1_channels, output_channels, kernel_size=3, padding=1),
        )
        self.deform_l2 = DeformConv2d(
            s1_channels,
            s1_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.refine_l2 = nn.Sequential(
            nn.Conv2d(s1_channels, s1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(s1_channels),
            nn.ReLU(inplace=True),
        )

        # Light cascade refinement at the final scale.
        self.offset_mask_cascade = nn.Sequential(
            nn.Conv2d(s1_channels * 2, s1_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1_channels, output_channels, kernel_size=3, padding=1),
        )
        self.deform_cascade = DeformConv2d(
            s1_channels,
            s1_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.refine_cascade = nn.Sequential(
            nn.Conv2d(s1_channels, s1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(s1_channels),
            nn.ReLU(inplace=True),
        )

    def _deform(self, feature, offset_mask, deform):
        offset_channels = 2 * self.kernel_size * self.kernel_size
        offset = offset_mask[:, :offset_channels]
        mask = offset_mask[:, offset_channels:].sigmoid()
        return deform(feature, offset, mask)

    def forward(self, neighbor_s2, center_s2, neighbor_s1, center_s1):
        # L3: coarse alignment at s2.
        offset_mask_l3 = self.offset_mask_l3(
            torch.cat([neighbor_s2, center_s2], dim=1)
        )
        aligned_s2 = self._deform(
            neighbor_s2,
            offset_mask_l3,
            self.deform_l3,
        )
        aligned_s2 = self.refine_l3(aligned_s2)

        # L2: use the coarse aligned feature as an explicit guide.
        coarse_s2 = F.interpolate(
            self.coarse_to_s1(aligned_s2),
            size=neighbor_s1.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        offset_mask_l2 = self.offset_mask_l2(
            torch.cat([neighbor_s1, center_s1, coarse_s2], dim=1)
        )
        aligned_s1 = self._deform(
            neighbor_s1,
            offset_mask_l2,
            self.deform_l2,
        )
        aligned_s1 = self.refine_l2(aligned_s1)

        # Cascade refinement at s1.
        offset_mask_cascade = self.offset_mask_cascade(
            torch.cat([aligned_s1, center_s1], dim=1)
        )
        aligned_s1 = self._deform(
            aligned_s1,
            offset_mask_cascade,
            self.deform_cascade,
        )
        aligned_s1 = self.refine_cascade(aligned_s1)
        return aligned_s2, aligned_s1


class TemporalGateFusion(nn.Module):
    """Center-guided TSA-style residual fusion for three aligned features."""

    def __init__(self, channels=128, embed_channels=32):
        super(TemporalGateFusion, self).__init__()
        self.channels = channels
        self.embed_channels = embed_channels

        self.query = nn.Conv2d(channels, embed_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, embed_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # Small but non-zero residual keeps the center path stable while
        # allowing the temporal branch to receive gradients immediately.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, center, previous, following):
        features = [previous, center, following]
        query = self.query(center)
        scale = math.sqrt(float(self.embed_channels))
        logits = []
        values = []
        for feature in features:
            key = self.key(feature)
            logits.append((query * key).sum(dim=1, keepdim=True) / scale)
            values.append(self.value(feature))

        weights = torch.softmax(torch.cat(logits, dim=1), dim=1)
        weighted = sum(
            value * weights[:, index:index + 1]
            for index, value in enumerate(values)
        )

        spatial_weight = self.spatial_attention(
            torch.cat([weighted, center], dim=1)
        )
        residual = self.fuse(
            torch.cat([weighted, center, weighted - center], dim=1)
        )
        residual = residual * spatial_weight
        return center + self.residual_scale * residual


class TemporalReliabilityGate(nn.Module):
    """Predict whether aligned temporal evidence should affect detection.

    The existing temporal fusion always produces a residual feature.  This
    module adds an explicit center-frame fallback: a reliability value of
    zero rejects the temporal residual, while a value of one preserves it.
    The gate is spatial rather than global so that reliable objects and
    unreliable background regions can be treated differently.
    """

    def __init__(self, channels=128, embed_channels=16, init_bias=2.2):
        super(TemporalReliabilityGate, self).__init__()
        self.project = nn.Conv2d(channels, embed_channels, kernel_size=1)
        self.predictor = nn.Sequential(
            nn.Conv2d(7, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, 1, kernel_size=1),
        )

        # Start close to the existing multi-frame behavior while keeping
        # enough sigmoid gradient for the gate to learn to reject residuals.
        nn.init.normal_(self.predictor[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.predictor[-1].bias, init_bias)

    def forward(self, center, previous, following, temporal):
        center_embed = F.normalize(self.project(center), dim=1, eps=1e-6)
        previous_embed = F.normalize(self.project(previous), dim=1, eps=1e-6)
        following_embed = F.normalize(self.project(following), dim=1, eps=1e-6)

        center_previous = (center_embed * previous_embed).sum(
            dim=1, keepdim=True
        )
        center_following = (center_embed * following_embed).sum(
            dim=1, keepdim=True
        )
        previous_following = (previous_embed * following_embed).sum(
            dim=1, keepdim=True
        )
        center_previous_diff = (center_embed - previous_embed).abs().mean(
            dim=1, keepdim=True
        )
        center_following_diff = (center_embed - following_embed).abs().mean(
            dim=1, keepdim=True
        )
        neighbor_diff = (previous_embed - following_embed).abs().mean(
            dim=1, keepdim=True
        )
        residual_magnitude = torch.tanh(
            (temporal - center).abs().mean(dim=1, keepdim=True)
        )

        gate_features = torch.cat([
            center_previous,
            center_following,
            previous_following,
            center_previous_diff,
            center_following_diff,
            neighbor_diff,
            residual_magnitude,
        ], dim=1)
        return torch.sigmoid(self.predictor(gate_features))


class TemporalDetectionHead(nn.Module):
    """Shared auxiliary heatmap head for the two aligned neighbor features."""

    def __init__(self, channels, num_classes, hidden_channels=64):
        super(TemporalDetectionHead, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        # The auxiliary head is newly initialized when loading a two-level
        # checkpoint. A low initial foreground prior keeps its focal loss
        # stable during the first fine-tuning steps.
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, feature):
        heatmap = self.heatmap(self.features(feature))
        return F.interpolate(
            heatmap,
            scale_factor=2,
            mode='bilinear',
            align_corners=False,
        )


class DREB_Net_MF(DREB_Net):
    """DREB with two-level EDVR-style temporal alignment and fusion."""

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
        # Shared by t-1 and t+1.  The first implementation intentionally
        # stops at s1 to control memory on a single 24GB GPU.
        self.temporal_align = TemporalDeformAlign(
            s2_channels=128,
            s1_channels=64,
        )
        self.temporal_fusion_s2 = TemporalGateFusion(channels=128)
        self.temporal_fusion_s1 = TemporalGateFusion(channels=64, embed_channels=16)
        # The s1 deformable convolutions operate at 1/4 resolution and are
        # the largest new activation block.  Checkpoint this temporal branch
        # during training even though the original CLI does not expose the
        # backbone's optional checkpoint flag.
        self.temporal_checkpoint = True

    def _run_stage(self, blocks, value):
        for block in blocks:
            if self.use_checkpoint and value.requires_grad:
                value = checkpoint.checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        return value

    def _align_neighbor(self, neighbor_s2, center_s2, neighbor_s1, center_s1):
        if self.temporal_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint.checkpoint(
                self.temporal_align,
                neighbor_s2,
                center_s2,
                neighbor_s1,
                center_s1,
                use_reentrant=False,
            )
        return self.temporal_align(
            neighbor_s2,
            center_s2,
            neighbor_s1,
            center_s1,
        )

    def _select_detection_temporal_feature(self, center, previous, following,
                                            temporal):
        """Keep the original two-level temporal feature on the base model."""
        return temporal

    def _temporal_training_outputs(self, previous_s2, following_s2):
        """Optional hook used only by separate auxiliary-supervision models."""
        return None

    def forward(self, x, mode):
        if x.ndim != 5:
            raise ValueError(
                'DREB_Net_MF expects [B, T, C, H, W], got {}'.format(tuple(x.shape))
            )
        if x.size(1) != 3 or x.size(2) != 3:
            raise ValueError('DREB_Net_MF expects exactly three RGB frames')

        batch_size, num_frames, channels, height, width = x.shape
        shallow = x.reshape(batch_size * num_frames, channels, height, width)
        shallow = self.stage0(shallow)
        s0_all = shallow
        s1_all = self.deblur_down1(s0_all)
        s2_all = self.deblur_down2(s1_all)

        s0_all = s0_all.reshape(
            batch_size, num_frames, s0_all.size(1), s0_all.size(2), s0_all.size(3)
        )
        s1_all = s1_all.reshape(
            batch_size, num_frames, s1_all.size(1), s1_all.size(2), s1_all.size(3)
        )
        s2_all = s2_all.reshape(
            batch_size, num_frames, s2_all.size(1), s2_all.size(2), s2_all.size(3)
        )

        s0 = s0_all[:, 1]
        s1 = s1_all[:, 1]
        s2 = s2_all[:, 1]

        previous_s2, previous_s1 = self._align_neighbor(
            s2_all[:, 0], s2, s1_all[:, 0], s1
        )
        following_s2, following_s1 = self._align_neighbor(
            s2_all[:, 2], s2, s1_all[:, 2], s1
        )

        s2_temporal_base = self.temporal_fusion_s2(
            s2, previous_s2, following_s2
        )
        s1_temporal = self.temporal_fusion_s1(
            s1, previous_s1, following_s1
        )
        s2_temporal_detect = self._select_detection_temporal_feature(
            s2, previous_s2, following_s2, s2_temporal_base
        )

        # The original two-level DREB detection path receives the temporal s2
        # feature. Subclasses may explicitly override the selection hook.
        out = self._run_stage(self.stage1, s0)
        out = self._run_stage(self.stage2, out)
        out_LFAMM = self.LFAMM(out)
        out = self.MAGFF_attention(out, s2_temporal_detect)
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
            temporal_outputs = self._temporal_training_outputs(
                previous_s2, following_s2
            )
            if temporal_outputs is not None:
                ret['temporal_hm'] = temporal_outputs
            # Keep the existing deblur loss and supervision unchanged.  Only
            # replace the feature sources: s2 and s1 now use temporal fusion,
            # while the high-resolution s0 skip remains center-frame-only.
            down3 = self.deblur_down3(s2_temporal_base)
            down4 = self.deblur_down4(down3)
            up1 = self.deblur_up1(down4, down3)
            up2 = self.deblur_up2(up1, s2_temporal_base)
            up3 = self.deblur_up3(up2, s1_temporal)
            up4 = self.deblur_up4(up3, s0)
            deblur_out = self.deblur_up5(up4, None)
            return [ret], deblur_out
        raise ValueError('mode not eq train/val!!!')


class DREB_Net_MF_RG(DREB_Net_MF):
    """Two-level DREB with a detection-only temporal reliability gate."""

    def __init__(self, *args, **kwargs):
        super(DREB_Net_MF_RG, self).__init__(*args, **kwargs)
        self.temporal_reliability_gate = TemporalReliabilityGate(
            channels=128,
            embed_channels=16,
            init_bias=2.2,
        )

    def _select_detection_temporal_feature(self, center, previous, following,
                                            temporal):
        reliability = self.temporal_reliability_gate(
            center, previous, following, temporal
        )
        # q=0 is an exact center-frame fallback; q=1 reproduces the existing
        # two-level temporal fusion output.
        return center + reliability * (temporal - center)


class DREB_Net_MF_TDS(DREB_Net_MF):
    """Independent B/C architecture with auxiliary temporal detection head."""

    def __init__(self, *args, **kwargs):
        super(DREB_Net_MF_TDS, self).__init__(*args, **kwargs)
        self.temporal_detection_head = TemporalDetectionHead(
            channels=128,
            num_classes=self.heads['hm'],
        )

    def _temporal_training_outputs(self, previous_s2, following_s2):
        # Both aligned neighbors use exactly the same auxiliary head.
        return torch.stack([
            self.temporal_detection_head(previous_s2),
            self.temporal_detection_head(following_s2),
        ], dim=1)


def create_DREB_Net_multiframe_detect(deploy=False, use_checkpoint=False, heads=None,
                                      head_conv=None):
    print('create_DREB_Net_MF')
    return DREB_Net_MF(
        override_groups_map=None,
        deploy=deploy,
        use_checkpoint=use_checkpoint,
        heads=heads,
        head_conv=head_conv,
    )


def create_DREB_Net_multiframe_reliability_detect(
        deploy=False, use_checkpoint=False, heads=None, head_conv=None):
    print('create_DREB_Net_MF_RG')
    return DREB_Net_MF_RG(
        override_groups_map=None,
        deploy=deploy,
        use_checkpoint=use_checkpoint,
        heads=heads,
        head_conv=head_conv,
    )


def create_DREB_Net_multiframe_tds_detect(
        deploy=False, use_checkpoint=False, heads=None, head_conv=None):
    print('create_DREB_Net_MF_TDS')
    return DREB_Net_MF_TDS(
        override_groups_map=None,
        deploy=deploy,
        use_checkpoint=use_checkpoint,
        heads=heads,
        head_conv=head_conv,
    )
