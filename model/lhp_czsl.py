"""
LHP-CZSL Stage 1: LLM-guided Hierarchical Prototype CZSL

ClusPro Baseline 대비 변경사항:
1. Variable K: primitive별 prototype 수를 LLM이 결정 (sub_meanings.json)
2. L_sem: 같은 primitive 내 prototype 간 의미 거리 정렬 손실
3. Cosine Decorrelation: HSIC 대체 (소배치 안정성)
4. (v3_text, optional) Text-side sub-meaning ensemble: text_ensemble=True 일 때
   primitive당 K_p sub-meaning 학습 임베딩을 두고, attr/obj head는 text feature
   레벨에서 mean-pool, comp head는 임베딩 레벨에서 mean-pool 후 prompt에 주입.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip_modules.clip_model import load_clip
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import CustomTextEncoder
from model.nce_loss import ContrastiveLoss


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64, dropout=0.0, adapter_scalar="0.1"):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = dropout
        self.scale = float(adapter_scalar)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        down = self.non_linear_func(self.down_proj(x))
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale
        return (up + residual) if add_residual else up


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        return F.dropout(F.relu(self.bn1_fc(self.fc1(x))), training=self.training)


def cosine_decorrelation(f_attr, f_obj):
    """Cosine decorrelation loss: f_attr ⊥ f_obj"""
    f_a = F.normalize(f_attr, dim=-1)
    f_o = F.normalize(f_obj, dim=-1)
    cos_sim = (f_a * f_o).sum(dim=-1)
    return cos_sim.abs().mean()


class LHPCZSL(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()

        clip_arch = config.clip_arch if hasattr(config, 'clip_arch') and config.clip_arch else config.clip_model
        self.clip = load_clip(name=clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset
        self.enable_pos_emb = True

        dtype = self.clip.dtype or torch.float16
        self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, dtype)

        # Freeze CLIP
        for p in self.parameters():
            p.requires_grad = False

        output_dim = self.clip.visual.output_dim

        # ---- Load LLM sub-meanings ----
        sub_meanings_path = getattr(config, 'sub_meanings_path', None)
        self.k_max = getattr(config, 'cluster_num', 5)
        self.k_min = getattr(config, 'cluster_min', 1)
        self.text_ensemble = getattr(config, 'text_ensemble', False)
        if self.text_ensemble and sub_meanings_path is None:
            raise ValueError("text_ensemble=True requires sub_meanings_path")
        (self.attr_k, self.obj_k,
         attr_sem_target, attr_sem_mask, obj_sem_target, obj_sem_mask,
         self.attr_sub_names_list, self.obj_sub_names_list) = \
            self._load_sub_meanings(sub_meanings_path, output_dim)
        # Register as buffers so they move with model.cuda()
        self.register_buffer('attr_sem_target', attr_sem_target)
        self.register_buffer('attr_sem_mask', attr_sem_mask)
        self.register_buffer('obj_sem_target', obj_sem_target)
        self.register_buffer('obj_sem_mask', obj_sem_mask)

        # ---- Sub-meaning indexing for text-ensemble ----
        # Always built (cheap); used only when text_ensemble=True.
        attr_sub_to_attr = []
        for i, subs in enumerate(self.attr_sub_names_list):
            attr_sub_to_attr.extend([i] * len(subs))
        obj_sub_to_obj = []
        for i, subs in enumerate(self.obj_sub_names_list):
            obj_sub_to_obj.extend([i] * len(subs))
        self.register_buffer('attr_sub_to_attr_idx', torch.tensor(attr_sub_to_attr, dtype=torch.long))
        self.register_buffer('obj_sub_to_obj_idx', torch.tensor(obj_sub_to_obj, dtype=torch.long))
        self.sum_Ka = len(attr_sub_to_attr)
        self.sum_Ko = len(obj_sub_to_obj)

        # ---- Visual Adapters ----
        num_blocks = self.clip.visual.transformer.layers
        vision_width = self.clip.visual.transformer.width
        adapter_dim = getattr(config, 'adapter_dim', 64)
        adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        self.visual_adapters = nn.ModuleList([
            Adapter(vision_width, adapter_dim, adapter_dropout)
            for _ in range(2 * num_blocks)
        ])

        # ---- Disentanglers ----
        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)
        self.attr_proj = Disentangler(output_dim)
        self.obj_proj = Disentangler(output_dim)

        # ---- Prototype Memory (padded to k_max, masked by attr_k/obj_k) ----
        self.momentum = getattr(config, 'proto_momentum', 0.99)

        for i in range(len(self.attributes)):
            self.register_buffer(f"attr_queue{i}", torch.randn(self.k_max, output_dim))
        for i in range(len(self.classes)):
            self.register_buffer(f"obj_queue{i}", torch.randn(self.k_max, output_dim))

        # ---- Contrastive Loss ----
        self.nceloss = ContrastiveLoss()
        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # ---- Single Soft Prompt ----
        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()
        self.soft_att_obj = nn.Parameter(base_soft_emb)
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # ---- Loss weights ----
        self.pair_loss_weight = getattr(config, 'pair_loss_weight', 1.0)
        self.attr_loss_weight = getattr(config, 'attr_loss_weight', 1.0)
        self.obj_loss_weight = getattr(config, 'obj_loss_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
        self.decorr_weight = getattr(config, 'decorr_weight', 0.1)
        self.sem_weight = getattr(config, 'sem_weight', 0.05)

        # ---- Inference weights ----
        self.pair_inf_w = getattr(config, 'pair_inference_weight', 1.0)
        self.attr_inf_w = getattr(config, 'attr_inference_weight', 1.0)
        self.obj_inf_w = getattr(config, 'obj_inference_weight', 1.0)

    # ==================================================================
    # LLM Sub-meanings Loading
    # ==================================================================

    def _load_sub_meanings(self, path, output_dim):
        """Load sub_meanings.json and build per-primitive K values, d_sem matrices,
        and per-primitive sub-meaning name lists.

        Sub-name lists are always populated. With no path or no entry for a primitive,
        defaults to K=1 with sub=[primitive_name] so the structure is uniform.
        """
        attr_k = [self.k_max] * len(self.attributes)
        obj_k = [self.k_max] * len(self.classes)

        # Pre-build batched d_sem target tensor: (num_prims, k_max, k_max)
        # and mask for valid pairs
        attr_sem_target = torch.zeros(len(self.attributes), self.k_max, self.k_max)
        attr_sem_mask = torch.zeros(len(self.attributes), self.k_max, self.k_max, dtype=torch.bool)
        obj_sem_target = torch.zeros(len(self.classes), self.k_max, self.k_max)
        obj_sem_mask = torch.zeros(len(self.classes), self.k_max, self.k_max, dtype=torch.bool)

        # Per-primitive sub-name lists (degenerate to [name] when no info available)
        attr_sub_names = [[name] for name in self.attributes]
        obj_sub_names = [[name] for name in self.classes]

        if path is None:
            return (attr_k, obj_k,
                    attr_sem_target, attr_sem_mask, obj_sem_target, obj_sem_mask,
                    attr_sub_names, obj_sub_names)

        with open(path, 'r') as f:
            data = json.load(f)

        attrs_data = data.get('attrs', {})
        objs_data = data.get('objs', {})

        # train.py normalizes primitive names (`.replace('.', ' ').lower()`)
        # before passing to the model, but JSON keys are typically the raw
        # dotted/cased dataset names (e.g. 'Faux.Fur'). Build a normalized
        # lookup so matching works regardless.
        def _norm(s):
            return s.replace('.', ' ').lower()
        attrs_data = {_norm(k): v for k, v in attrs_data.items()}
        objs_data = {_norm(k): v for k, v in objs_data.items()}

        for i, attr_name in enumerate(self.attributes):
            if attr_name in attrs_data:
                info = attrs_data[attr_name]
                k = max(min(info['K'], self.k_max), self.k_min)
                attr_k[i] = k
                sub_names = info.get('sub', [attr_name])[:k]
                if len(sub_names) < k:
                    sub_names = sub_names + [attr_name] * (k - len(sub_names))
                attr_sub_names[i] = sub_names
                if k >= 2 and 'd_sem' in info:
                    for pair_key, dist in info['d_sem'].items():
                        parts = pair_key.split('-', 1)
                        if len(parts) == 2 and parts[0] in sub_names and parts[1] in sub_names:
                            idx_a = sub_names.index(parts[0])
                            idx_b = sub_names.index(parts[1])
                            # Store as similarity target (1 - distance)
                            attr_sem_target[i, idx_a, idx_b] = 1.0 - dist
                            attr_sem_target[i, idx_b, idx_a] = 1.0 - dist
                            attr_sem_mask[i, idx_a, idx_b] = True
                            attr_sem_mask[i, idx_b, idx_a] = True

        for i, obj_name in enumerate(self.classes):
            if obj_name in objs_data:
                info = objs_data[obj_name]
                k = max(min(info['K'], self.k_max), self.k_min)
                obj_k[i] = k
                sub_names = info.get('sub', [obj_name])[:k]
                if len(sub_names) < k:
                    sub_names = sub_names + [obj_name] * (k - len(sub_names))
                obj_sub_names[i] = sub_names
                if k >= 2 and 'd_sem' in info:
                    for pair_key, dist in info['d_sem'].items():
                        parts = pair_key.split('-', 1)
                        if len(parts) == 2 and parts[0] in sub_names and parts[1] in sub_names:
                            idx_a = sub_names.index(parts[0])
                            idx_b = sub_names.index(parts[1])
                            obj_sem_target[i, idx_a, idx_b] = 1.0 - dist
                            obj_sem_target[i, idx_b, idx_a] = 1.0 - dist
                            obj_sem_mask[i, idx_a, idx_b] = True
                            obj_sem_mask[i, idx_b, idx_a] = True

        sum_ka = sum(len(s) for s in attr_sub_names)
        sum_ko = sum(len(s) for s in obj_sub_names)
        n_attr_kgt1 = sum(1 for s in attr_sub_names if len(s) > 1)
        n_obj_kgt1 = sum(1 for s in obj_sub_names if len(s) > 1)
        print(f"[_load_sub_meanings] sub_meanings_path={path} | "
              f"attrs: total={len(self.attributes)}, K>1={n_attr_kgt1}, sum_K={sum_ka} | "
              f"objs: total={len(self.classes)}, K>1={n_obj_kgt1}, sum_K={sum_ko}")
        if self.text_ensemble and (n_attr_kgt1 + n_obj_kgt1) == 0:
            raise RuntimeError(
                "text_ensemble=True but no primitive matched a K>1 entry in "
                f"{path}. Check key normalization (train.py lowercases and "
                "replaces '.' with ' ' before passing names to the model)."
            )

        return (attr_k, obj_k,
                attr_sem_target, attr_sem_mask, obj_sem_target, obj_sem_mask,
                attr_sub_names, obj_sub_names)

    @staticmethod
    def _format_sub_name(sub_name, primitive_name):
        """Clean LLM-derived sub names (e.g. 'smooth_leather' -> 'smooth leather').
        Leave names that match the primitive name untouched to preserve baseline-equivalent
        tokenization for K=1 primitives (e.g. 'Hair.Calf' stays 'Hair.Calf')."""
        if sub_name == primitive_name:
            return sub_name
        return sub_name.replace('_', ' ').replace('.', ' ').strip()

    # ==================================================================
    # Image Encoding (identical to ClusPro baseline)
    # ==================================================================

    def encode_image(self, x):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        num_blocks = self.clip.visual.transformer.layers
        for i in range(num_blocks):
            block = self.clip.visual.transformer.resblocks[i]
            adapt_x = self.visual_adapters[i](x, add_residual=False)
            residual = x
            x = block.attention(block.ln_1(x))
            x = x + adapt_x + residual
            adapt_x = self.visual_adapters[i + num_blocks](x, add_residual=False)
            residual = x
            x = block.mlp(block.ln_2(x))
            x = x + adapt_x + residual

        x = x.permute(1, 0, 2)
        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x[:, 0, :], x

    # ==================================================================
    # Soft Prompt (identical to ClusPro baseline)
    # ==================================================================

    def _construct_soft_prompt(self):
        prompt_template = self.config.prompt_template
        ctx_init = self.config.ctx_init

        token_ids = self.tokenizer(
            prompt_template, context_length=self.config.context_length
        ).cuda()

        if self.text_ensemble:
            # Sub-level: one learnable embedding per (primitive, sub-meaning), in order.
            # Init = mean of CLIP token embeddings of the sub-meaning name (cleaned).
            attr_sub_strings = [
                self._format_sub_name(s, prim)
                for prim, subs in zip(self.attributes, self.attr_sub_names_list)
                for s in subs
            ]
            obj_sub_strings = [
                self._format_sub_name(s, prim)
                for prim, subs in zip(self.classes, self.obj_sub_names_list)
                for s in subs
            ]
            tokens_to_init = attr_sub_strings + obj_sub_strings
        else:
            tokens_to_init = self.attributes + self.classes

        tokenized = torch.cat([
            self.tokenizer(tok, context_length=self.config.context_length)
            for tok in tokens_to_init
        ])
        orig_emb = self.clip.token_embedding(tokenized.cuda())
        soft_emb = torch.zeros(len(tokens_to_init), orig_emb.size(-1))
        for idx, rep in enumerate(orig_emb):
            eos_idx = tokenized[idx].argmax()
            soft_emb[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx = embedding[0, 1:1+n_ctx[0], :].to(self.clip.dtype)
        attr_ctx = embedding[1, 1:1+n_ctx[1], :].to(self.clip.dtype)
        obj_ctx = embedding[2, 1:1+n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_emb, comp_ctx, attr_ctx, obj_ctx

    def _pool_sub_features(self, features, sub_to_prim, num_prims):
        """Mean-pool sub-meaning features back to per-primitive features.

        features: (S, D), sub_to_prim: (S,) long, num_prims: int.
        Returns: (num_prims, D) with mean over each primitive's sub-features.
        """
        D = features.shape[-1]
        idx = sub_to_prim.unsqueeze(-1).expand(-1, D)
        out = torch.zeros(num_prims, D, device=features.device, dtype=features.dtype)
        out.scatter_add_(0, idx, features)
        ones = torch.ones(features.shape[0], device=features.device, dtype=features.dtype)
        counts = torch.zeros(num_prims, device=features.device, dtype=features.dtype)
        counts.scatter_add_(0, sub_to_prim, ones)
        return out / counts.unsqueeze(-1).clamp(min=1.0)

    def _construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]

        if self.text_ensemble:
            num_elements = [len(pair_idx), self.sum_Ka, self.sum_Ko]
        else:
            num_elements = [len(pair_idx), self.offset, len(self.classes)]
        token_tensor = []

        for i in range(self.token_ids.shape[0]):
            ids = self.token_ids[i].repeat(num_elements[i], 1)
            token_tensor.append(
                self.clip.token_embedding(ids.cuda()).type(self.clip.dtype)
            )

        eos_idx = [int(self.token_ids[i].argmax()) for i in range(self.token_ids.shape[0])]
        embs = self.attr_dropout(self.soft_att_obj)

        if self.text_ensemble:
            attr_subs = embs[:self.sum_Ka]
            obj_subs = embs[self.sum_Ka:]
            # comp prompt: pool sub-emb to per-primitive then index
            attr_pooled = self._pool_sub_features(
                attr_subs, self.attr_sub_to_attr_idx, len(self.attributes)
            )
            obj_pooled = self._pool_sub_features(
                obj_subs, self.obj_sub_to_obj_idx, len(self.classes)
            )
            token_tensor[0][:, eos_idx[0]-2, :] = attr_pooled[attr_idx].type(self.clip.dtype)
            token_tensor[0][:, eos_idx[0]-1, :] = obj_pooled[obj_idx].type(self.clip.dtype)
            token_tensor[0][:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

            # attr-only / obj-only: sub-level prompts (one per sub-meaning)
            token_tensor[1][:, eos_idx[1]-1, :] = attr_subs.type(self.clip.dtype)
            token_tensor[1][:, 1:len(self.attr_ctx)+1, :] = self.attr_ctx.type(self.clip.dtype)
            token_tensor[2][:, eos_idx[2]-1, :] = obj_subs.type(self.clip.dtype)
            token_tensor[2][:, 1:len(self.obj_ctx)+1, :] = self.obj_ctx.type(self.clip.dtype)
        else:
            token_tensor[0][:, eos_idx[0]-2, :] = embs[attr_idx].type(self.clip.dtype)
            token_tensor[0][:, eos_idx[0]-1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
            token_tensor[0][:, 1:len(self.comp_ctx)+1, :] = self.comp_ctx.type(self.clip.dtype)

            token_tensor[1][:, eos_idx[1]-1, :] = embs[:self.offset].type(self.clip.dtype)
            token_tensor[1][:, 1:len(self.attr_ctx)+1, :] = self.attr_ctx.type(self.clip.dtype)

            token_tensor[2][:, eos_idx[2]-1, :] = embs[self.offset:].type(self.clip.dtype)
            token_tensor[2][:, 1:len(self.obj_ctx)+1, :] = self.obj_ctx.type(self.clip.dtype)

        return token_tensor

    # ==================================================================
    # Prototype Update (variable K via masking)
    # ==================================================================

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def _update_prototypes(self, batch_attr, batch_obj, attr_idx, obj_idx):
        batch_attr_f = batch_attr.float()
        batch_obj_f = batch_obj.float()

        if not (torch.isfinite(batch_attr_f).all() and torch.isfinite(batch_obj_f).all()):
            return

        # Hard one-hot cluster assignment (matches official ClusPro train_forward, ICLR 2025).
        # Soft softmax assignment (previous version) collapses prototypes since every sample
        # contributes to every cluster — see RESEARCH_LOG §5.3.
        for k in range(len(self.attributes)):
            mask = (attr_idx == k)
            if mask.sum() == 0:
                continue
            K_k = self.attr_k[k]
            feats_k = batch_attr_f[mask]
            queue_k = getattr(self, f"attr_queue{k}")
            queue_f = queue_k.float()
            # Only use first K_k prototypes
            sim = l2_normalize(feats_k) @ l2_normalize(queue_f[:K_k]).t()
            couplings = F.softmax(sim / 0.5, dim=-1)
            if K_k > 1:
                assign = F.gumbel_softmax(couplings, tau=0.5, hard=True, dim=-1)
            else:
                assign = couplings  # K=1: hard assignment is trivial (always cluster 0)
            new_proto = assign.t() @ feats_k
            counts = assign.sum(dim=0)
            valid = counts > 0
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[:K_k][valid] = queue_f[:K_k][valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k[:K_k] = l2_normalize(queue_f[:K_k]).to(queue_k.dtype)

        for k in range(len(self.classes)):
            mask = (obj_idx == k)
            if mask.sum() == 0:
                continue
            K_k = self.obj_k[k]
            feats_k = batch_obj_f[mask]
            queue_k = getattr(self, f"obj_queue{k}")
            queue_f = queue_k.float()
            sim = l2_normalize(feats_k) @ l2_normalize(queue_f[:K_k]).t()
            couplings = F.softmax(sim / 0.5, dim=-1)
            if K_k > 1:
                assign = F.gumbel_softmax(couplings, tau=0.5, hard=True, dim=-1)
            else:
                assign = couplings
            new_proto = assign.t() @ feats_k
            counts = assign.sum(dim=0)
            valid = counts > 0
            if valid.any():
                new_proto[valid] = l2_normalize(new_proto[valid])
                queue_f[:K_k][valid] = queue_f[:K_k][valid] * self.momentum + new_proto[valid] * (1 - self.momentum)
            queue_k[:K_k] = l2_normalize(queue_f[:K_k]).to(queue_k.dtype)

    def _get_cluster_labels(self, batch_feat, idx_labels, primitives, prim_type):
        B = batch_feat.shape[0]
        k_list = self.attr_k if prim_type == "attr" else self.obj_k
        labels = torch.zeros(B, dtype=torch.long, device=batch_feat.device)
        for k in range(len(primitives)):
            mask = (idx_labels == k)
            if mask.sum() == 0:
                continue
            K_k = k_list[k]
            queue_k = getattr(self, f"{prim_type}_queue{k}")
            # Only compare against valid prototypes
            sim = l2_normalize(batch_feat[mask]) @ l2_normalize(queue_k[:K_k]).t()
            cluster_idx = sim.argmax(dim=-1)
            labels[mask] = cluster_idx + k * self.k_max
        return labels

    def _gather_all_prototypes(self, prim_type, primitives):
        protos = []
        for k in range(len(primitives)):
            protos.append(getattr(self, f"{prim_type}_queue{k}"))
        return torch.stack(protos, dim=0)

    # ==================================================================
    # L_sem: Intra-Primitive Semantic Distance Alignment
    # ==================================================================

    def compute_l_sem(self):
        """
        L_sem = Σ_p Σ_{i,j ∈ sub(p)} |sim(p_i, p_j) - (1 - d_sem(s_i, s_j))|²
        Batched computation: no Python loops over primitives.
        """
        loss = torch.tensor(0.0, device=self.attr_sem_target.device)
        count = 0

        for prim_type, sem_target, sem_mask in [
            ("attr", self.attr_sem_target, self.attr_sem_mask),
            ("obj", self.obj_sem_target, self.obj_sem_mask),
        ]:
            if not sem_mask.any():
                continue
            primitives = self.attributes if prim_type == "attr" else self.classes
            # Gather all prototypes: (N, k_max, d)
            all_protos = self._gather_all_prototypes(prim_type, primitives)
            all_protos_norm = l2_normalize(all_protos.float())
            # Batched pairwise sim: (N, k_max, k_max)
            vis_sim = torch.bmm(all_protos_norm, all_protos_norm.transpose(1, 2))
            # Masked MSE
            diff = (vis_sim - sem_target) ** 2
            loss = loss + diff[sem_mask].sum()
            count += sem_mask.sum().item()

        return loss / max(count, 1)

    # ==================================================================
    # Forward
    # ==================================================================

    def train_forward(self, batch, idx):
        batch_img = batch[0].cuda()
        attr_idx, obj_idx = batch[1], batch[2]

        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))
        B = f_global.shape[0]

        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)

        self._update_prototypes(f_attr, f_obj, attr_idx, obj_idx)

        attr_labels = self._get_cluster_labels(f_attr, attr_idx, self.attributes, "attr")
        obj_labels = self._get_cluster_labels(f_obj, obj_idx, self.classes, "obj")

        attr_protos = self._gather_all_prototypes("attr", self.attributes)
        obj_protos = self._gather_all_prototypes("obj", self.classes)

        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        all_protos_flat = torch.cat([
            self.attr_proj(attr_protos.reshape(-1, attr_protos.shape[-1])).reshape(-1, attr_protos.shape[-1]),
            self.obj_proj(obj_protos.reshape(-1, obj_protos.shape[-1])).reshape(-1, obj_protos.shape[-1]),
        ], dim=0)

        protos_expanded = all_protos_flat.unsqueeze(0).expand(B, -1, -1)

        loss_contrastive = (
            self.nceloss(f_attr_proj, protos_expanded, attr_labels) +
            self.nceloss(f_obj_proj, protos_expanded, obj_labels)
        )

        # Cosine decorrelation (replaces HSIC)
        loss_decorr = cosine_decorrelation(f_attr_proj, f_obj_proj)

        # L_sem
        loss_sem = self.compute_l_sem()

        # ---- Text encoding ----
        token_tensors = self._construct_token_tensors(idx)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        logits = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb
            )
            if self.text_ensemble and i == 1:
                feat = self._pool_sub_features(
                    feat, self.attr_sub_to_attr_idx, len(self.attributes)
                )
            elif self.text_ensemble and i == 2:
                feat = self._pool_sub_features(
                    feat, self.obj_sub_to_obj_idx, len(self.classes)
                )
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * norm_img[i] @ feat.t())

        comp_logits, attr_logits, obj_logits = logits
        return comp_logits, attr_logits, obj_logits, loss_contrastive, loss_decorr, loss_sem

    def val_forward(self, batch, idx):
        batch_img = batch[0].cuda()
        f_global, _ = self.encode_image(batch_img.type(self.clip.dtype))

        f_attr = self.attr_disentangler(f_global)
        f_obj = self.obj_disentangler(f_global)
        f_attr_proj = self.attr_proj(f_attr)
        f_obj_proj = self.obj_proj(f_obj)

        token_tensors = self._construct_token_tensors(idx)

        img_feats = [f_global, f_attr_proj, f_obj_proj]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in img_feats]
        logit_scale = self.clip.logit_scale.exp()

        logits = []
        for i in range(self.token_ids.shape[0]):
            feat, _ = self.text_encoder(
                self.token_ids[i], token_tensors[i], enable_pos_emb=self.enable_pos_emb
            )
            if self.text_ensemble and i == 1:
                feat = self._pool_sub_features(
                    feat, self.attr_sub_to_attr_idx, len(self.attributes)
                )
            elif self.text_ensemble and i == 2:
                feat = self._pool_sub_features(
                    feat, self.obj_sub_to_obj_idx, len(self.classes)
                )
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * norm_img[i] @ feat.t())

        return logits[0], logits[1], logits[2]

    # ==================================================================
    # Loss & Inference
    # ==================================================================

    def loss_calu(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        if self.training:
            comp_logits, attr_logits, obj_logits, loss_contras, loss_decorr, loss_sem = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        loss = (
            self.pair_loss_weight * loss_fn(comp_logits, batch_target) +
            self.attr_loss_weight * loss_fn(attr_logits, batch_attr) +
            self.obj_loss_weight * loss_fn(obj_logits, batch_obj)
        )

        if self.training:
            loss = (loss
                    + self.contrastive_weight * loss_contras
                    + self.decorr_weight * loss_decorr
                    + self.sem_weight * loss_sem)

        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)

        for i in range(comp_logits.shape[-1]):
            w_attr = 1 if self.attr_inf_w == 0 else attr_pred[:, pairs[i][0]] * self.attr_inf_w
            w_obj = 1 if self.obj_inf_w == 0 else obj_pred[:, pairs[i][1]] * self.obj_inf_w
            comp_logits[:, i] = comp_logits[:, i] * self.pair_inf_w + w_attr * w_obj

        return comp_logits

    def forward(self, batch, idx):
        if self.training:
            return self.train_forward(batch, idx)
        else:
            with torch.no_grad():
                return self.val_forward(batch, idx)
