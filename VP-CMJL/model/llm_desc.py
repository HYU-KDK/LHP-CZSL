"""LLM description manager + Phase 3 helpers for VP-CMJL.

Two consumers:
  - Phase 2 / AD-CA: DescriptionManager picks per-attribute (or per-object) LLM
    embedding contextualized to the current batch's dominant peer (e.g. for
    AD-CA, peer = dominant object in batch; for OD-CA, peer = dominant attr).
  - Phase 3 / Visual proxy: load_proxy_init returns alpha-blend init tensors;
    load_unseen_alignment returns (pair_idx, target_emb) for L_unseen.

All buffers are kept in float32 on the model's device; the model casts to
self.clip.dtype at use site.
"""
import os
import torch
import torch.nn as nn


class DescriptionManager(nn.Module):
    """Looks up contextual LLM embeddings keyed by a peer index.

    For AD-CA: input emb is [N_attr, N_obj, D] and peer = dominant obj index.
    For OD-CA: input emb is [N_attr, N_obj, D] but transposed view; peer = dominant attr.

    Usage:
        attr_dm = DescriptionManager(emb_b, mask_b, attr_fallback, axis="row")
        attr_llm = attr_dm.get_context(batch_obj_idx)  # [N_attr, D]
    """
    def __init__(self, emb, mask, fallback, axis="row"):
        super().__init__()
        # emb: [N_a, N_o, D]; mask: [N_a, N_o] bool
        # axis="row": fixed axis 0 (attrs), peer index over axis 1 (objs)
        # axis="col": fixed axis 1 (objs), peer index over axis 0 (attrs)
        assert axis in ("row", "col")
        self.axis = axis
        self.register_buffer("emb", emb.float(), persistent=False)
        self.register_buffer("mask", mask.bool(), persistent=False)
        # fallback shape: [N_axis, D] where N_axis = emb.shape[0 if row else 1]
        self.register_buffer("fallback", fallback.float(), persistent=False)

    def get_context(self, peer_indices):
        """peer_indices: [B] long tensor of peer-side indices in current batch.

        Returns [N_axis, D] tensor: for each fixed-axis entry, the embedding
        contextualized to the dominant peer in this batch (mode of peer_indices),
        falling back to per-row mean when that (row, dom_peer) cell has no desc.
        """
        # Determine dominant peer
        if peer_indices.numel() == 0:
            # degenerate: no batch — return fallback
            return self.fallback
        N_peer = self.emb.shape[1] if self.axis == "row" else self.emb.shape[0]
        counts = torch.bincount(peer_indices.long(), minlength=N_peer)
        dom = int(counts.argmax().item())

        if self.axis == "row":
            slab = self.emb[:, dom, :]      # [N_a, D]
            m = self.mask[:, dom]           # [N_a]
        else:
            slab = self.emb[dom, :, :]      # [N_o, D]
            m = self.mask[dom, :]           # [N_o]

        out = torch.where(m.unsqueeze(-1), slab, self.fallback)
        return out


def load_attr_obj_managers(desc_pt_path):
    """Load attr_in_context_emb.pt and build (attr_dm, obj_dm).

    attr_dm: input emb [N_a, N_o, D]; peer index = obj_idx; output = [N_a, D]
    obj_dm:  input emb (same), but axis transposed; peer index = attr_idx; output = [N_o, D]
    """
    if not os.path.exists(desc_pt_path):
        raise FileNotFoundError(desc_pt_path)
    d = torch.load(desc_pt_path, map_location="cpu", weights_only=False)
    emb = d["emb"]                    # [N_a, N_o, D]
    mask = d["mask"]                  # [N_a, N_o]
    attr_fallback = d["attr_fallback"]  # [N_a, D]
    # obj fallback: per-obj mean over attrs that have desc
    N_a, N_o, D = emb.shape
    obj_fallback = torch.zeros(N_o, D)
    for oi in range(N_o):
        m = mask[:, oi]
        if m.any():
            obj_fallback[oi] = emb[m, oi].mean(dim=0)
    attr_dm = DescriptionManager(emb, mask, attr_fallback, axis="row")
    obj_dm = DescriptionManager(emb, mask, obj_fallback, axis="col")
    return attr_dm, obj_dm


def load_proxy_init(proxy_init_pt_path, n_attr, n_obj, dim):
    """Returns (attr_init [N_a, D], obj_init [N_o, D]) tensors. Validates shapes."""
    d = torch.load(proxy_init_pt_path, map_location="cpu", weights_only=False)
    attr_init = d["attr_init"].float()
    obj_init = d["obj_init"].float()
    assert attr_init.shape == (n_attr, dim), \
        f"attr_init shape {tuple(attr_init.shape)} != ({n_attr},{dim})"
    assert obj_init.shape == (n_obj, dim), \
        f"obj_init shape {tuple(obj_init.shape)} != ({n_obj},{dim})"
    return attr_init, obj_init


def load_unseen_alignment(unseen_pt_path):
    """Returns (pair_idx [N_unseen, 2], target_emb [N_unseen, D]) on CPU float32."""
    d = torch.load(unseen_pt_path, map_location="cpu", weights_only=False)
    return d["pair_idx"].long(), d["emb"].float()
