"""LitePT teacher/student self-distillation following Sonata's three losses.

Adapted from the official Sonata model at Pointcept commit
96e109d103418a5daaa375c50e1665b9f04f4a4b. Our CPU augmentation pipeline
already produces masked global views. Teacher BatchNorm uses EMA running
statistics in evaluation mode, an explicit adaptation for LitePT's BN layers.
"""

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn

from cluster import OnlineCluster
from registry import MODELS
from .losses import matched_distillation_loss, up_cast


@MODELS.register_module(name=["SSLModel", "SonataLitePT"])
class SSLModel(nn.Module):
    """Own teacher/student backbones, separate heads and spatial SSL losses.

    Packed views are scene-major: [scene0/global0, scene0/global1, ...].
    Masked globals compare to the corresponding and the other teacher global;
    all locals compare to global0 from their own scene. No point matching ever
    crosses scene boundaries. Only student modules participate in autograd.
    """

    def __init__(
        self,
        backbone,
        head_in_channels=None,
        head_hidden_channels=4096,
        head_embed_channels=256,
        head_num_prototypes=4096,
        up_cast_level=2,
        num_global_view=2,
        num_local_view=4,
        match_max_r=0.32,
        student_temp=0.1,
        mask_loss_weight=0.25,
        roll_mask_loss_weight=0.25,
        unmask_loss_weight=0.5,
        teacher_custom=None,
        sinkhorn_iterations=3,
        match_query_chunk_size=1024,
        match_reference_chunk_size=4096,
        backbone_amp=True,
    ):
        super().__init__()
        if num_global_view not in (1, 2):
            raise ValueError("Sonata pairing supports one or two global views")
        if roll_mask_loss_weight > 0 and num_global_view != 2:
            raise ValueError("rolled masked loss requires two global views")
        if unmask_loss_weight > 0 and num_local_view < 1:
            raise ValueError("unmasked local loss requires at least one local view")
        weights = (mask_loss_weight, roll_mask_loss_weight, unmask_loss_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("loss weights must be nonnegative with at least one enabled")
        if up_cast_level < 0 or student_temp <= 0 or match_max_r <= 0:
            raise ValueError("invalid upcast level, student temperature or matching radius")
        self.up_cast_level = up_cast_level
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view
        self.match_max_r = match_max_r
        self.student_temp = student_temp
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_query_chunk_size = match_query_chunk_size
        self.match_reference_chunk_size = match_reference_chunk_size
        self.backbone_amp = backbone_amp

        if isinstance(backbone, Mapping):
            # Lazy import keeps the loss/head utilities usable without SpConv,
            # FlashAttention, or registration of a particular backbone.
            from builder import build_model

            config = deepcopy(dict(backbone))
            config.setdefault("enc_mode", True)
            if not config["enc_mode"]:
                raise ValueError("SSLModel requires an encoder-only LitePT backbone")
            channels = config.get("enc_channels", (36, 72, 144, 252, 504))
            if up_cast_level >= len(channels):
                raise ValueError("up_cast_level exceeds the encoder pooling hierarchy")
            if head_in_channels is None:
                head_in_channels = sum(channels[-(up_cast_level + 1) :])
            student_backbone = build_model(config)
            teacher_config = deepcopy(config)
            teacher_config.update(teacher_custom or {})
            teacher_backbone = build_model(teacher_config)
        elif isinstance(backbone, nn.Module):
            if teacher_custom:
                raise ValueError("teacher_custom requires a backbone configuration dictionary")
            if hasattr(backbone, "enc_mode") and not backbone.enc_mode:
                raise ValueError("SSLModel requires an encoder-only LitePT backbone")
            if head_in_channels is None:
                raise ValueError("head_in_channels is required for a backbone module")
            student_backbone = backbone
            teacher_backbone = deepcopy(backbone)
        else:
            raise TypeError("backbone must be an nn.Module or model configuration")

        student_modules = {"backbone": student_backbone}
        teacher_modules = {"backbone": teacher_backbone}

        def head():
            return OnlineCluster(
                in_ch=head_in_channels,
                hidd_ch=head_hidden_channels,
                em_ch=head_embed_channels,
                num_prototypes=head_num_prototypes,
            )

        if mask_loss_weight > 0 or roll_mask_loss_weight > 0:
            student_modules["mask_head"] = head()
            teacher_modules["mask_head"] = head()
        if unmask_loss_weight > 0:
            student_modules["unmask_head"] = head()
            teacher_modules["unmask_head"] = head()
        self.student = nn.ModuleDict(student_modules)
        self.teacher = nn.ModuleDict(teacher_modules)
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        # Unlike PTv3's LayerNorm-only backbone, LitePT has BatchNorm. Running
        # it in eval mode removes teacher dropout and independent BN updates;
        # update_teacher below transfers the student's running statistics.
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA after a successful optimizer step, including BN running state."""
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be between zero and one")
        student_parameters = dict(self.student.named_parameters())
        for name, teacher_parameter in self.teacher.named_parameters():
            teacher_parameter.mul_(momentum).add_(student_parameters[name], alpha=1 - momentum)
        student_buffers = dict(self.student.named_buffers())
        for name, teacher_buffer in self.teacher.named_buffers():
            student_buffer = student_buffers[name]
            if teacher_buffer.is_floating_point() or teacher_buffer.is_complex():
                teacher_buffer.mul_(momentum).add_(student_buffer, alpha=1 - momentum)
            else:
                teacher_buffer.copy_(student_buffer)

    def _backbone(self, branch, packed):
        # The default follows the trainer's AMP context. Keeping this explicit
        # override allows FP32-backbone experiments with AMP projection heads.
        if self.backbone_amp:
            point = branch["backbone"](dict(packed))
        else:
            with torch.autocast(device_type=packed["feat"].device.type, enabled=False):
                point = branch["backbone"](dict(packed))
        return up_cast(point, self.up_cast_level)

    def _loss(self, student_scores, teacher_scores, student_point, teacher_point, pairing, temp):
        return matched_distillation_loss(
            student_scores,
            teacher_scores,
            student_point,
            teacher_point,
            pairing,
            teacher_temp=temp,
            student_temp=self.student_temp,
            match_max_r=self.match_max_r,
            sinkhorn_iterations=self.sinkhorn_iterations,
            query_chunk_size=self.match_query_chunk_size,
            reference_chunk_size=self.match_reference_chunk_size,
        )

    def forward(self, batch, teacher_temp=0.07):
        if teacher_temp <= 0:
            raise ValueError("teacher_temp must be positive")
        view_counts = batch.get("view_counts")
        if view_counts is not None and (
            view_counts["global"] != self.num_global_view
            or view_counts["local"] != self.num_local_view
        ):
            raise ValueError("augmentation and SSLModel view counts differ")
        with torch.no_grad():
            teacher_point = self._backbone(self.teacher, batch["global"])
        count = len(teacher_point["offset"])
        if count == 0 or count % self.num_global_view:
            raise ValueError("global output offsets do not contain complete scene view groups")
        batch_size = count // self.num_global_view
        zero = next(self.student.parameters()).sum() * 0.0
        result = {"loss": zero}
        for name in ("mask", "roll_mask", "unmask"):
            result[name + "_loss"] = zero
            result[name + "_matched"] = zero.detach().new_zeros((), dtype=torch.long)
            result[name + "_matched_views"] = zero.detach().new_zeros((), dtype=torch.long)

        def record(name, value, weight):
            result[name + "_loss"] = value.loss
            result[name + "_matched"] = value.matched
            result[name + "_matched_views"] = value.matched_views
            result["loss"] = result["loss"] + weight * value.loss

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            if batch.get("masked_global") is None or "mask" not in batch["masked_global"]:
                raise ValueError("masked_global must include a boolean input mask")
            student_point = self._backbone(self.student, batch["masked_global"])
            if len(student_point["offset"]) != count:
                raise ValueError("masked and teacher global view counts differ")
            student_scores = self.student["mask_head"](student_point["feat"])
            with torch.no_grad():
                teacher_scores = self.teacher["mask_head"](teacher_point["feat"])
            if self.mask_loss_weight > 0:
                value = self._loss(
                    student_scores, teacher_scores, student_point, teacher_point,
                    list(range(count)), teacher_temp,
                )
                record("mask", value, self.mask_loss_weight)
            if self.roll_mask_loss_weight > 0:
                # Swap the two globals within each scene, never between scenes.
                pairing = [view ^ 1 for view in range(count)]
                value = self._loss(
                    student_scores, teacher_scores, student_point, teacher_point,
                    pairing, teacher_temp,
                )
                record("roll_mask", value, self.roll_mask_loss_weight)

        if self.unmask_loss_weight > 0:
            if batch.get("local") is None:
                raise ValueError("local views are required for unmask_loss")
            student_point = self._backbone(self.student, batch["local"])
            if len(student_point["offset"]) != batch_size * self.num_local_view:
                raise ValueError("local output offsets do not match the configured scene/view count")
            student_scores = self.student["unmask_head"](student_point["feat"])
            with torch.no_grad():
                teacher_scores = self.teacher["unmask_head"](teacher_point["feat"])
            pairing = [
                (view // self.num_local_view) * self.num_global_view
                for view in range(batch_size * self.num_local_view)
            ]
            value = self._loss(
                student_scores, teacher_scores, student_point, teacher_point,
                pairing, teacher_temp,
            )
            record("unmask", value, self.unmask_loss_weight)
        return result
