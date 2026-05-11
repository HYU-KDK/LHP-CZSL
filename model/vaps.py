"""
VAPS: Visual Adaptive Prompting System (Stein et al., arXiv:2502.20292).

Implements the paper's three components on top of frozen CLIP ViT-L/14:
  - Visual prompt repository with similarity-based retrieval (top-2 prompts,
    averaged to form f_ret).
  - CoCoOp-style text prompt adapter (PromptNet) that shifts the comp-prompt
    prefix tokens per image.
  - Decomposition + cross-attention fusion producing f_{t->v}, scored against
    f_ret in a shared pair space (L_ret).

Loss = L_ret + lambda_attr_obj * (L_att + L_obj) + lambda_sp * L_sp.

Note: per-image prefix shift means the comp text encoder is run once per
sample in the batch. With small B (e.g. 8) and small |C_seen| this is OK on
UT-Zappos; for MIT-States / C-GQA consider increasing time budget.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_modules.clip_model import load_clip
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import CustomTextEncoder


class PromptNet(nn.Module):
    """CoCoOp-style adapter: f_v -> bias for prefix tokens."""
    def __init__(self, in_dim, ctx_dim, n_ctx, hidden=128):
        super().__init__()
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_ctx * ctx_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f_v):
        return self.net(f_v).view(-1, self.n_ctx, self.ctx_dim)


class VAPS(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_arch = getattr(config, 'clip_arch', None) or config.clip_model
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

        D = self.clip.visual.output_dim
        D_ctx = self.clip.token_embedding.weight.shape[1]
        self.D = D
        self.D_ctx = D_ctx

        # --- Soft prompts (comp / attr / obj), same structure as cluspro_baseline ---
        self.token_ids, base_soft_emb, self.comp_ctx, self.attr_ctx, self.obj_ctx = \
            self._construct_soft_prompt()
        self.soft_att_obj = nn.Parameter(base_soft_emb)
        self.comp_ctx = nn.Parameter(self.comp_ctx)
        self.attr_ctx = nn.Parameter(self.attr_ctx)
        self.obj_ctx = nn.Parameter(self.obj_ctx)

        # --- Text prompt adapter (per-image bias for comp prefix) ---
        n_ctx = self.comp_ctx.shape[0]
        self.prompt_adapter = PromptNet(D, D_ctx, n_ctx, hidden=getattr(config, 'adapter_hidden', 128))

        # --- Visual prompt repository ---
        self.M = getattr(config, 'repo_size', 20)
        self.top_n = getattr(config, 'top_n_prompts', 2)
        self.visual_prompts = nn.Parameter(torch.randn(self.M, D) * 0.02)
        self.prompt_keys = nn.Parameter(torch.randn(self.M, D) * 0.02)

        # --- Decomposition + cross-attention fusion (Q from f_t, K=V from f_v) ---
        self.cross_attn = nn.MultiheadAttention(D, num_heads=getattr(config, 'fusion_heads', 4),
                                                batch_first=True)
        # Pair space projection
        self.proj_fused = nn.Linear(D, D)
        self.proj_ret = nn.Linear(D, D)

        self.attr_dropout = nn.Dropout(getattr(config, 'attr_dropout', 0.3))

        # --- Loss weights ---
        self.lambda_attr_obj = getattr(config, 'lambda_attr_obj', 1.0)
        self.lambda_sp = getattr(config, 'lambda_sp', 1.0)
        self.lambda_ret = getattr(config, 'lambda_ret', 1.0)

        # --- Inference weights ---
        self.pair_inf_w = getattr(config, 'pair_inference_weight', 1.0)
        self.attr_inf_w = getattr(config, 'attr_inference_weight', 1.0)
        self.obj_inf_w = getattr(config, 'obj_inference_weight', 1.0)

    # ------------------------------------------------------------------
    # Image encoding (vanilla frozen CLIP visual; returns CLS feature + tokens)
    # ------------------------------------------------------------------
    def encode_image(self, x):
        v = self.clip.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)
        x = v.ln_post(x)
        if v.proj is not None:
            x = x @ v.proj
        return x[:, 0, :], x  # (B,D), (B,L,D)

    # ------------------------------------------------------------------
    # Soft prompt construction (mirrors cluspro_baseline)
    # ------------------------------------------------------------------
    def _construct_soft_prompt(self):
        prompt_template = self.config.prompt_template
        ctx_init = self.config.ctx_init

        token_ids = self.tokenizer(
            prompt_template, context_length=self.config.context_length
        ).cuda()

        tokenized = torch.cat([
            self.tokenizer(tok, context_length=self.config.context_length)
            for tok in self.attributes + self.classes
        ])
        orig_emb = self.clip.token_embedding(tokenized.cuda())
        soft_emb = torch.zeros(
            len(self.attributes) + len(self.classes), orig_emb.size(-1)
        )
        for idx, rep in enumerate(orig_emb):
            eos_idx = tokenized[idx].argmax()
            soft_emb[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx = embedding[0, 1:1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx = embedding[1, 1:1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx = embedding[2, 1:1 + n_ctx[2], :].to(self.clip.dtype)
        return token_ids, soft_emb, comp_ctx, attr_ctx, obj_ctx

    def _build_token_tensors(self, pair_idx, comp_ctx_override=None):
        """Build token tensors for comp/attr/obj heads.

        If comp_ctx_override is given with shape (n_ctx, D_ctx), use it for the
        comp prompt prefix (per-image shift uses this).
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        num_elements = [len(pair_idx), self.offset, len(self.classes)]
        token_tensor = []
        for i in range(self.token_ids.shape[0]):
            ids = self.token_ids[i].repeat(num_elements[i], 1)
            token_tensor.append(
                self.clip.token_embedding(ids.cuda()).type(self.clip.dtype)
            )
        eos_idx = [int(self.token_ids[i].argmax()) for i in range(self.token_ids.shape[0])]
        embs = self.attr_dropout(self.soft_att_obj)

        comp_ctx_use = self.comp_ctx if comp_ctx_override is None else comp_ctx_override

        token_tensor[0][:, eos_idx[0] - 2, :] = embs[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = embs[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(comp_ctx_use) + 1, :] = comp_ctx_use.type(self.clip.dtype)

        token_tensor[1][:, eos_idx[1] - 1, :] = embs[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx) + 1, :] = self.attr_ctx.type(self.clip.dtype)

        token_tensor[2][:, eos_idx[2] - 1, :] = embs[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx) + 1, :] = self.obj_ctx.type(self.clip.dtype)
        return token_tensor

    # ------------------------------------------------------------------
    # Visual prompt repository: top-N retrieval over keys
    # ------------------------------------------------------------------
    def _retrieve_prompts(self, f_v):
        # f_v: (B, D)
        fv_n = F.normalize(f_v.float(), dim=-1)
        keys_n = F.normalize(self.prompt_keys.float(), dim=-1)
        sim = fv_n @ keys_n.t()  # (B, M)
        topk = sim.topk(self.top_n, dim=-1)
        idx = topk.indices  # (B, top_n)
        selected = self.visual_prompts[idx]  # (B, top_n, D)
        f_ret = selected.mean(dim=1)  # (B, D)
        return f_ret, sim

    # ------------------------------------------------------------------
    # Cross-attention fusion: Q from f_t, K=V from f_v patches.
    # ------------------------------------------------------------------
    def _fuse(self, f_t, f_v_tokens):
        """f_t: (B, P, D) or (P, D); f_v_tokens: (B, L, D). Returns (B, P, D)."""
        if f_t.dim() == 2:
            f_t = f_t.unsqueeze(0).expand(f_v_tokens.shape[0], -1, -1)
        out, _ = self.cross_attn(f_t.float(), f_v_tokens.float(), f_v_tokens.float())
        return out

    # ------------------------------------------------------------------
    # Encode text heads
    # ------------------------------------------------------------------
    def _encode_text(self, token_tensor, head_idx):
        feat, _ = self.text_encoder(
            self.token_ids[head_idx], token_tensor, enable_pos_emb=self.enable_pos_emb
        )
        return feat

    def _comp_text_features(self, pair_idx, f_v):
        """Per-image shifted comp text features.

        Returns f_t_comp of shape (B, P, D) where P = |pair_idx|.
        """
        B = f_v.shape[0]
        bias = self.prompt_adapter(f_v.float()).type(self.clip.dtype)  # (B, n_ctx, D_ctx)
        feats = []
        for b in range(B):
            comp_ctx_b = self.comp_ctx + bias[b]
            token_tensor = self._build_token_tensors(pair_idx, comp_ctx_override=comp_ctx_b)
            f_t_b = self._encode_text(token_tensor[0], head_idx=0)  # (P, D)
            feats.append(f_t_b)
        return torch.stack(feats, dim=0)  # (B, P, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def train_forward(self, batch, pair_idx):
        img = batch[0].cuda()
        attr_idx_b, obj_idx_b = batch[1].cuda(), batch[2].cuda()
        target = batch[3].cuda()

        f_v, f_v_tokens = self.encode_image(img.type(self.clip.dtype))  # (B,D), (B,L,D)
        B = f_v.shape[0]
        logit_scale = self.clip.logit_scale.exp()

        # ---- Attribute / Object heads (no per-image shift) ----
        token_tensor = self._build_token_tensors(pair_idx)
        f_t_comp_global = self._encode_text(token_tensor[0], 0)  # (P,D)
        f_t_attr = self._encode_text(token_tensor[1], 1)  # (|A|,D)
        f_t_obj = self._encode_text(token_tensor[2], 2)  # (|O|,D)

        fv_n = f_v / f_v.norm(dim=-1, keepdim=True)
        fa_n = f_t_attr / f_t_attr.norm(dim=-1, keepdim=True)
        fo_n = f_t_obj / f_t_obj.norm(dim=-1, keepdim=True)
        attr_logits = logit_scale * fv_n @ fa_n.t()
        obj_logits = logit_scale * fv_n @ fo_n.t()

        # ---- Comp head with per-image prefix shift ----
        f_t_comp = self._comp_text_features(pair_idx, f_v)  # (B,P,D)
        f_t_comp_n = f_t_comp / f_t_comp.norm(dim=-1, keepdim=True)
        comp_logits = logit_scale * (fv_n.unsqueeze(1) * f_t_comp_n).sum(dim=-1)  # (B,P)

        # ---- Visual prompt retrieval ----
        f_ret, _ = self._retrieve_prompts(f_v)  # (B, D)

        # ---- Cross-attention fusion ----
        # Use the per-image comp text features as Q.
        f_tv = self._fuse(f_t_comp, f_v_tokens)  # (B,P,D)
        f_tv = self.proj_fused(f_tv)
        f_ret_proj = self.proj_ret(f_ret)  # (B,D)

        # Retrieval logits: <f_ret, f_{t->v}_pair>
        ret_logits = (f_ret_proj.unsqueeze(1) * f_tv).sum(dim=-1)  # (B,P)
        ret_logits = ret_logits * logit_scale

        return comp_logits, attr_logits, obj_logits, ret_logits

    def val_forward(self, batch, pair_idx):
        img = batch[0].cuda()
        f_v, _ = self.encode_image(img.type(self.clip.dtype))
        logit_scale = self.clip.logit_scale.exp()

        token_tensor = self._build_token_tensors(pair_idx)
        f_t_attr = self._encode_text(token_tensor[1], 1)
        f_t_obj = self._encode_text(token_tensor[2], 2)

        fv_n = f_v / f_v.norm(dim=-1, keepdim=True)
        fa_n = f_t_attr / f_t_attr.norm(dim=-1, keepdim=True)
        fo_n = f_t_obj / f_t_obj.norm(dim=-1, keepdim=True)
        attr_logits = logit_scale * fv_n @ fa_n.t()
        obj_logits = logit_scale * fv_n @ fo_n.t()

        f_t_comp = self._comp_text_features(pair_idx, f_v)  # (B,P,D)
        f_t_comp_n = f_t_comp / f_t_comp.norm(dim=-1, keepdim=True)
        comp_logits = logit_scale * (fv_n.unsqueeze(1) * f_t_comp_n).sum(dim=-1)

        return comp_logits, attr_logits, obj_logits

    # ------------------------------------------------------------------
    # Loss and inference adapters (compatible with train.py / test.py)
    # ------------------------------------------------------------------
    def loss_calu(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        batch_attr, batch_obj, batch_target = target[1].cuda(), target[2].cuda(), target[3].cuda()
        if self.training:
            comp_logits, attr_logits, obj_logits, ret_logits = predict
        else:
            comp_logits, attr_logits, obj_logits = predict

        L_sp = loss_fn(comp_logits, batch_target)
        L_att = loss_fn(attr_logits, batch_attr)
        L_obj = loss_fn(obj_logits, batch_obj)
        loss = self.lambda_sp * L_sp + self.lambda_attr_obj * (L_att + L_obj)
        if self.training:
            L_ret = loss_fn(ret_logits, batch_target)
            loss = loss + self.lambda_ret * L_ret
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
        with torch.no_grad():
            return self.val_forward(batch, idx)
