# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from .common import LayerNorm2d
from einops import rearrange

class FrequancyEncoding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(self,in_tensor):
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

    def forward(self, in_tensor):
        return self.pytorch_fwd(in_tensor)

class PosEncodingNeRFFixed(nn.Module):
    def __init__(self, in_channel, frequencies=10):
        super().__init__()

        self.in_channel = in_channel
        self.frequencies = frequencies
        self.n_output_dims = in_channel + 2 * in_channel * frequencies

    def get_out_dim(self) -> int:
        return self.n_output_dims

    def forward(self, coords):
        b,h,w,c = coords.shape
        coords = coords.view(coords.shape[0], -1, self.in_channel)
        coords_pos_enc = coords
        for i in range(self.frequencies):
            for j in range(self.in_channel):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * math.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.n_output_dims).squeeze(1).reshape(b,h,w,-1)

class SegField(nn.Module):
    def __init__(self,
                features: int=256, 
                cls_channel: int=1,
                coord_normalized_min: float=-1.0,
                coord_normalized_max: float=1.0,
                coords_channel: int=3,
                w0: int=10,
                top_k_ratio: float=0.125):
        super().__init__()

        self.coord_normalized_min = coord_normalized_min
        self.coord_normalized_max = coord_normalized_max
        self.cls_channel = cls_channel
        self.out_channel = 1
        self.pos_enc = FrequancyEncoding(in_dim=coords_channel, num_frequencies=w0, min_freq_exp=0.0, max_freq_exp=w0, include_input=True)

        self.top_k_ratio = top_k_ratio

        input_dim = 32 + 256  + self.pos_enc.get_out_dim()
        fine_input_dim = features + 32 + 256 + self.pos_enc.get_out_dim()
        self.seg_net = nn.ModuleList()
        self.seg_net.append(
            nn.Sequential(
                nn.Linear(input_dim, features*4), nn.BatchNorm1d(features*4), nn.ReLU(),
                nn.Linear(features*4, features*2), nn.BatchNorm1d(features*2), nn.ReLU(),
            )
        )

        self.drop_out = nn.Dropout(0.5)
        self.seg_net.append(
            nn.Sequential(
                nn.Linear(fine_input_dim, features), nn.BatchNorm1d(features), nn.ReLU(),
                nn.Linear(features, features), nn.BatchNorm1d(features), nn.ReLU(),
            )
        )
        self.seg_net.append(
            nn.ModuleList([nn.Sequential(nn.Linear(features*2, features + self.out_channel)) for _ in range(self.cls_channel)])
        )
        self.seg_net.append(
            nn.ModuleList([nn.Sequential(nn.Linear(features, features//2), nn.BatchNorm1d(features//2), nn.ReLU(), nn.Linear(features//2, self.out_channel)) for _ in range(self.cls_channel)])
        )
        for net in self.seg_net:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, image_embedding, image_pe, original_shape, epoch_T=1.0, cls=0):
        b, _, h, w = image_pe.shape
        h, w = original_shape

        d = 1
        coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(self.coord_normalized_min, self.coord_normalized_max, d),
                torch.linspace(
                    self.coord_normalized_min, self.coord_normalized_max, h
                ),
                torch.linspace(self.coord_normalized_min, self.coord_normalized_max, w),
                indexing="ij",
            ),
            axis=-1,
        ).to(image_embedding.device)

        coordinates = torch.repeat_interleave(coordinates, image_embedding.shape[0], dim=0)
        coordinates = self.pos_enc(coordinates)

        coordinates = rearrange(coordinates, 'b h w c -> (b h w) c')
        mask_coordinates = min(int(epoch_T*(coordinates.shape[-1]))+3, coordinates.shape[-1])
        coordinates[:, mask_coordinates:] = 0

        image_embedding = rearrange(image_embedding, 'b (h w) c -> b c h w', b=image_pe.shape[0], h=int(math.sqrt(image_embedding.shape[1])), w=int(math.sqrt(image_embedding.shape[1])))
        
        image_embedding = F.interpolate(image_embedding, size=original_shape, mode='bilinear', align_corners=False)
        image_pe = F.interpolate(image_pe, size=original_shape, mode='bilinear', align_corners=False)

        image_embedding_flatten = rearrange(image_embedding, 'b c h w -> (b h w) c')
        image_pe_flatten = rearrange(image_pe, 'b c h w -> (b h w) c')
        feat_concat = torch.cat([image_embedding_flatten, image_pe_flatten, coordinates], dim=1)
        seg_res = []

        seg_feat_concat = []
        for _ in range(2):
            seg_feat_ = self.seg_net[0](feat_concat)

            seg_feat_ = self.seg_net[2][cls](seg_feat_)

            seg_feat_ = self.drop_out(seg_feat_)
            seg_feat_concat.append(seg_feat_)
        seg_feat_concat = torch.stack(seg_feat_concat, dim=1)

        seg_feat = seg_feat_
        seg_feat_var = torch.var(seg_feat_concat, dim=1).mean(dim=1, keepdim=True)

        seg_coarse = seg_feat[:,0:self.out_channel]
        seg_fine = seg_coarse.clone()
        seg_feat = seg_feat[:,self.out_channel:]
        seg_coarse = rearrange(seg_coarse, '(b h w) c -> (h w) b c', b=b, h=h, w=w)
        seg_coarse = seg_coarse.transpose(1, 2).contiguous().view(b, self.out_channel, h, w)
        seg_res.append(seg_coarse)

        _, sel_ind = torch.topk(seg_feat_var[:,0], k=int(seg_feat_var.shape[0] * self.top_k_ratio))
        fine_feat_concat = torch.cat([feat_concat[sel_ind,...], seg_feat[sel_ind,...]], dim=1)
        seg_fine_sel = self.seg_net[1](fine_feat_concat)

        seg_fine[sel_ind,...] = self.seg_net[3][cls](seg_fine_sel)[:,0:self.out_channel]
        
        seg_fine = rearrange(seg_fine, '(b h w) c -> b (h w) c', b=b, h=h, w=w)
        seg_fine = seg_fine.transpose(1, 2).view(b, self.out_channel, h, w)
        seg_res.append(seg_fine)

        return seg_res

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        class_num: int = 1,
        top_k_ratio: float = 0.125,
        wo_inr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.wo_inr = wo_inr
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.seg_field = SegField(256, cls_channel=class_num, top_k_ratio=top_k_ratio)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks_field(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        original_size: Tuple[int, ...],
        epoch_T: float = 1.0,
        cls: torch.Tensor = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        if not self.wo_inr:
            hs, src = self.transformer(src, pos_src, tokens)
            src = src.transpose(1, 2).view(b, c, h, w)
            src = self.output_upscaling(src)
            src = src.view(b, src.shape[1], -1).transpose(1, 2)
            masks = self.seg_field(src, pos_src, original_size, epoch_T, cls)
        else:
            hs, src = self.transformer(src, pos_src, tokens)
            mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
            # Upscale mask embeddings and predict masks using the mask tokens
            src = src.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.output_upscaling(src)
            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            hyper_in = torch.stack(hyper_in_list, dim=1)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
