import math
import os
join = os.path.join

import torch
import torch.nn as nn
from segment_anything.modeling import Sam


class LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            q_lora_a: nn.Module,
            q_lora_b: nn.Module,
            k_lora_a: nn.Module,
            k_lora_b: nn.Module,
            v_lora_a: nn.Module,
            v_lora_b: nn.Module,
            w_gate: nn.Module,
            mode: str = "qv",
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.w_gate = w_gate
        self.mode = mode
        self.q_lora_a = q_lora_a
        self.q_lora_b = q_lora_b
        self.k_lora_a = k_lora_a
        self.k_lora_b = k_lora_b
        self.v_lora_a = v_lora_a
        self.v_lora_b = v_lora_b

    def forward(self, x):
        qkv = self.qkv(x)
        
        f = x.mean(dim=1).mean(dim=1)
        g = self.w_gate(f)
        
        if 'q' in self.mode:
            q_lora = [self.q_lora_b[0](self.q_lora_a[0](x))]
            q_lora = torch.stack(q_lora, dim=1)
            q_lora = torch.einsum('bemnc,be->bemnc', q_lora, g).sum(dim=1)
            qkv[:, :, :, :self.dim] += q_lora
        if 'k' in self.mode:
            k_lora = [self.k_lora_b[0](self.k_lora_a[0](x))]
            k_lora = torch.stack(k_lora, dim=1)
            k_lora = torch.einsum('bemnc,be->bemnc', k_lora, g).sum(dim=1)
            qkv[:, :, :, self.dim:2*self.dim] += k_lora
        if 'v' in self.mode:
            v_lora = [self.v_lora_b[0](self.v_lora_a[0](x))]
            v_lora = torch.stack(v_lora, dim=1)
            v_lora = torch.einsum('bemnc,be->bemnc', v_lora, g).sum(dim=1)
            qkv[:, :, :, -self.dim:] += v_lora
        
        return qkv

class IMedSAM(nn.Module):
    """Applies low-rank adaptation to a SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        r: rank of LoRA
        pos: which layer to apply LoRA
    """

    def __init__(self, sam: Sam, r: int=4, mode: str='qv', pos=None):
        super(IMedSAM, self).__init__()

        self.r = r
        self.mode = mode
        
        # assign LoRA layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))
        
        # freeze SAM image encoder and prompt encoder
        for name, param in sam.image_encoder.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = True
        for name, param in sam.mask_decoder.named_parameters():
            if "seg_field" in name or "output_upscaling" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # create LoRA layers for storage, then we can init them or load weights
        self.w_As = []
        self.w_Bs = []
        self.w_Gs = []
        
        # apply LoRA to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue
            
            qkv = blk.attn.qkv
            self.dim = qkv.in_features
            
            if 'q' in mode:
                q_lora_a = [nn.Linear(self.dim, r, bias=False)]
                q_lora_b = [nn.Linear(r, self.dim, bias=False)]
                self.w_As += q_lora_a
                self.w_Bs += q_lora_b
            else:
                q_lora_a = None
                q_lora_b = None
            if 'k' in mode:
                k_lora_a = [nn.Linear(self.dim, r, bias=False)]
                k_lora_b = [nn.Linear(r, self.dim, bias=False)]
                self.w_As += k_lora_a
                self.w_Bs += k_lora_b
            else:
                k_lora_a = None
                k_lora_b = None
            if 'v' in mode:
                v_lora_a = [nn.Linear(self.dim, r, bias=False)]
                v_lora_b = [nn.Linear(r, self.dim, bias=False)]
                self.w_As += v_lora_a
                self.w_Bs += v_lora_b
            else:
                v_lora_a = None
                v_lora_b = None

            gate = nn.Linear(self.dim, 1, bias=False)
            self.w_Gs.append(gate)
            
            blk.attn.qkv = LoRA_qkv(
                qkv,
                nn.ModuleList(q_lora_a),
                nn.ModuleList(q_lora_b),
                nn.ModuleList(k_lora_a),
                nn.ModuleList(k_lora_b),
                nn.ModuleList(v_lora_a),
                nn.ModuleList(v_lora_b),
                gate,
                mode=mode,
            )
        
        # init LoRA layer parameters
        self.reset_parameters()
        self.sam = sam
        
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self) -> dict:
        r"""save both lora and mask decoder parameters.
        """

        # save lora parameters
        num_lora_weight = len(self.w_As)
        num_gate_weight = len(self.w_Gs)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_lora_weight)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_lora_weight)}
        g_tensors = {f"w_g_{i:03d}": self.w_Gs[i].weight for i in range(num_gate_weight)}
        
        # save mask decoder parameters
        mask_decoder_tensors = {}
        image_freq_tensors = {}
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'freq' in key:
                image_freq_tensors[key] = value
            if 'prompt_encoder' in key:
                image_freq_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **g_tensors, **mask_decoder_tensors, **image_freq_tensors}
        return merged_dict

    def load_lora_parameters(self, state_dict) -> None:
        r"""load both lora and mask decoder parameters.
        """

        # load lora parameters
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.Parameter(saved_tensor)
        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.Parameter(saved_tensor)
        for i, w_G_linear in enumerate(self.w_Gs):
            saved_key = f"w_g_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_G_linear.weight = nn.Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load mask decoder parameters
        image_freq_keys = [k for k in sam_keys if 'freq' in k]
        image_freq_values = [state_dict[k] for k in image_freq_keys]
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]

        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]

        image_freq_new_state_dict = {k: v for k, v in zip(image_freq_keys, image_freq_values)}
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        
        sam_dict.update(mask_decoder_new_state_dict)
        sam_dict.update(image_freq_new_state_dict)

        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)
        
        self.sam.load_state_dict(sam_dict)

    def forward(self, img, box_torch=None, epoch_T=1.0, original_size=[], cls=0):
        
        # LoRA image encoder
        input_image = self.sam.preprocess(img) # (1, 3, 1024, 1024)
        image_embedding = self.sam.image_encoder(input_image) # (1, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None
        )

        if len(original_size) <= 0:
            original_size = img.shape[2:4]
        mask_predictions = self.sam.mask_decoder.predict_masks_field(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            original_size=original_size,
            epoch_T=epoch_T,
            cls=cls
        )
        return mask_predictions
