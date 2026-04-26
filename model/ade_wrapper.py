"""
ADE (Attention as Disentangler) wrapper for the baseline framework.
Adapts the ADE model interface to match forward(), loss_calu(), logit_infer().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from model.ade_common import MLP
from model.multi_head_attention import CrossAttention
from model.word_embedding import load_word_embeddings
from model.emd_utils import emd_inference_opencv_test
from model.image_extractor import get_image_extractor


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DsetProxy:
    """Minimal dataset-like object that ADE model expects."""
    pass


class ADEWrapper(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.config = config
        self.attributes = attributes  # lowercase attr names
        self.classes = classes        # lowercase obj names
        self.offset = offset

        # Build dset proxy (ADE expects dset with attrs, objs, pairs, etc.)
        # These will be set externally after dataset is loaded
        self.dset_proxy = None

        # Config defaults
        emb_dim = getattr(config, 'emb_dim', 300)
        feat_dim = 768  # DINO ViT-B/16
        cosine_scale = getattr(config, 'cosine_scale', 50)
        nlayers = getattr(config, 'nlayers', 2)
        fc_emb_str = getattr(config, 'fc_emb', '768,1024')
        dropout = getattr(config, 'dropout', True)
        norm = getattr(config, 'norm', True)
        relu = getattr(config, 'relu', False)
        emb_init = getattr(config, 'emb_init', 'glove')
        static_inp = getattr(config, 'static_inp', False)
        self.aow = getattr(config, 'aow', 1.0)
        self.scale = cosine_scale
        self.emb_dim = emb_dim

        # Parse fc_emb layers
        if isinstance(fc_emb_str, str):
            layers = [int(x) for x in fc_emb_str.split(',')]
        else:
            layers = [int(fc_emb_str)]

        # Image embedders (3 branches)
        self.image_embedder = MLP(feat_dim, emb_dim, relu=relu, num_layers=nlayers,
                                  dropout=dropout, norm=norm, layers=list(layers))
        self.image_embedder_obj = MLP(feat_dim, emb_dim, relu=relu, num_layers=nlayers,
                                      dropout=dropout, norm=norm, layers=list(layers))
        self.image_embedder_attr = MLP(feat_dim, emb_dim, relu=relu, num_layers=nlayers,
                                       dropout=dropout, norm=norm, layers=list(layers))

        # Cross-attention modules (DINO = 768 dim, 12 heads)
        self.cross_attn = CrossAttention(768, 12)
        self.cross_attn_attr = CrossAttention(768, 12)
        self.cross_attn_obj = CrossAttention(768, 12)

        # Semantic embeddings
        num_attrs = len(attributes)
        num_objs = len(classes)
        self.attr_embedder = nn.Embedding(num_attrs, emb_dim)
        self.obj_embedder = nn.Embedding(num_objs, emb_dim)

        # Init with word embeddings
        if emb_init:
            # Use original (non-lowercased) names for word embedding lookup
            pretrained_weight = load_word_embeddings(emb_init, attributes)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(emb_init, classes)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        if static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Projection for composing attr+obj
        self.projection = nn.Linear(emb_dim * 2, emb_dim)

        # Image extractor (DINO, frozen)
        self.image_extractor = get_image_extractor(arch='dino')
        self.image_extractor.eval()
        for param in self.image_extractor.parameters():
            param.requires_grad = False

    def setup_dset_proxy(self, dataset):
        """Call this after dataset is loaded to set up pair mappings."""
        proxy = DsetProxy()
        proxy.attrs = dataset.attrs
        proxy.objs = dataset.objs
        proxy.pairs = dataset.pairs
        proxy.train_pairs = dataset.train_pairs
        proxy.attr2idx = dataset.attr2idx
        proxy.obj2idx = dataset.obj2idx
        proxy.pair2idx = dataset.pair2idx
        proxy.feat_dim = 768
        proxy.open_world = getattr(dataset, 'open_world', False)
        self.dset_proxy = proxy

        # Precompute pair indices
        self.val_attrs = torch.LongTensor([dataset.attr2idx[a] for a, _ in dataset.pairs]).to(device)
        self.val_objs = torch.LongTensor([dataset.obj2idx[o] for _, o in dataset.pairs]).to(device)

    def extract_features(self, img):
        """Extract DINO features from raw images."""
        with torch.no_grad():
            feat = self.image_extractor(img)
        return feat

    def compose_word_embeddings(self, pairs_idx):
        """Compose attribute-object text embeddings."""
        attrs = self.attr_embedder(pairs_idx[:, 0])
        objs = self.obj_embedder(pairs_idx[:, 1])
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        return output

    def cos_logits(self, emb, proto):
        return torch.matmul(F.normalize(emb, dim=-1),
                            F.normalize(proto, dim=-1).permute(1, 0))

    def compute_EMD_loss(self, x, y):
        """EMD loss between attention maps."""
        def avg_pooling(attn):
            b, n_token, _ = attn.shape
            n_patch = n_token - 1
            h = int(n_patch ** 0.5)
            attn1 = attn[:, :, 1:]
            cls1 = attn[:, :, 0].unsqueeze(-1)
            pos1 = attn1.reshape(b, -1, h, h)
            pos1 = F.avg_pool2d(pos1, kernel_size=2, stride=2)
            attn_new = torch.cat([cls1, pos1.reshape(b, n_token, -1)], dim=-1).transpose(1, 2)
            attn2 = attn_new[:, :, 1:]
            cls2 = attn_new[:, :, 0].unsqueeze(-1)
            pos2 = attn2.reshape(b, -1, h, h)
            pos2 = F.avg_pool2d(pos2, kernel_size=2, stride=2)
            attn_new = torch.cat([cls2, pos2.reshape(b, -1, int((h / 2) ** 2))], dim=-1).transpose(1, 2)
            return attn_new

        if x.shape[-1] > 100:
            x = avg_pooling(x)
            y = avg_pooling(y)

        w_y = x[:, 0, 1:]
        w_x = y[:, 0, 1:]
        w_y = w_y / w_y.sum(-1).unsqueeze(-1)
        w_x = w_x / w_x.sum(-1).unsqueeze(-1)
        mat = (x[:, 1:, 1:] + y[:, 1:, 1:].transpose(1, 2)) / 2.
        _, flow = emd_inference_opencv_test(1 - mat, w_x, w_y)
        score = (flow * mat).sum(-1).sum(-1)
        return score.mean()

    def forward(self, batch, pairs_idx):
        """
        Args:
            batch: list from DataLoader
                Training: [img, attr, obj, pair, same_attr_img, ..., same_obj_img, ...]
                Eval: [img, attr, obj, pair]
            pairs_idx: [num_pairs, 2] tensor of (attr_idx, obj_idx)
        Returns:
            Training: (loss, None, emd_scores)
            Eval: (scores_dict,) - single-element tuple
        """
        if self.training:
            return self._train_forward(batch, pairs_idx)
        else:
            with torch.no_grad():
                return self._val_forward(batch, pairs_idx)

    def _train_forward(self, batch, pairs_idx):
        img = batch[0].cuda()
        attrs = batch[1].cuda()
        objs = batch[2].cuda()
        pairs = batch[3].cuda()

        # same_prim_sample data:
        # batch[4]=same_attr_img, batch[5]=same_attr_attr_idx, batch[6]=diff_obj_idx,
        # batch[7]=same_attr_pair_idx, batch[8]=same_attr_mask,
        # batch[9]=same_obj_img, batch[10]=diff_attr_idx, batch[11]=same_obj_obj_idx,
        # batch[12]=same_obj_pair_idx, batch[13]=same_obj_mask
        same_attr_img = batch[4].cuda()
        same_obj_img = batch[9].cuda()
        same_attr_mask = batch[8].cuda().float()
        same_obj_mask = batch[13].cuda().float()

        # Extract DINO features
        img_feat = self.extract_features(img)
        same_attr_feat = self.extract_features(same_attr_img)
        same_obj_feat = self.extract_features(same_obj_img)

        # Cross-attention for object disentanglement
        # same_obj_feat = image with same object, different attribute
        img_attn_obj, attn_obj = self.cross_attn_obj(q=img_feat, k=same_obj_feat, return_attention=True)
        img_attn_obj_bar, attn_obj_bar = self.cross_attn_obj(q=same_obj_feat, k=img_feat, return_attention=True)

        _, attn_diff_obj = self.cross_attn_obj(q=img_feat, k=same_attr_feat, return_attention=True)
        _, attn_diff_obj_bar = self.cross_attn_obj(q=same_attr_feat, k=img_feat, return_attention=True)

        # Cross-attention for attribute disentanglement
        # same_attr_feat = image with same attribute, different object
        img_attn_attr, attn_attr = self.cross_attn_attr(q=img_feat, k=same_attr_feat, return_attention=True)
        img_attn_attr_bar, attn_attr_bar = self.cross_attn_attr(q=same_attr_feat, k=img_feat, return_attention=True)

        _, attn_diff_attr = self.cross_attn_attr(q=img_feat, k=same_obj_feat, return_attention=True)
        _, attn_diff_attr_bar = self.cross_attn_attr(q=same_obj_feat, k=img_feat, return_attention=True)

        # Sum over heads
        attn_obj = attn_obj.sum(1)
        attn_obj_bar = attn_obj_bar.sum(1)
        attn_attr = attn_attr.sum(1)
        attn_attr_bar = attn_attr_bar.sum(1)
        attn_diff_obj = attn_diff_obj.sum(1)
        attn_diff_obj_bar = attn_diff_obj_bar.sum(1)
        attn_diff_attr = attn_diff_attr.sum(1)
        attn_diff_attr_bar = attn_diff_attr_bar.sum(1)

        # EMD losses
        score_obj = self.compute_EMD_loss(attn_obj, attn_obj_bar)
        score_attr = self.compute_EMD_loss(attn_attr, attn_attr_bar)
        score_diff_attr = self.compute_EMD_loss(attn_diff_attr, attn_diff_attr_bar)
        score_diff_obj = self.compute_EMD_loss(attn_diff_obj, attn_diff_obj_bar)
        score_dis = score_diff_attr + score_diff_obj
        loss_emd = score_dis - (score_attr + score_obj)

        # Composition attention
        img_attn = self.cross_attn(q=img_feat, k=img_feat)

        # Embed [CLS] tokens
        img_attn_obj_emb = self.image_embedder_obj(img_attn_obj[:, 0, :])
        img_attn_obj_bar_emb = self.image_embedder_obj(img_attn_obj_bar[:, 0, :])
        img_attn_attr_emb = self.image_embedder_attr(img_attn_attr[:, 0, :])
        img_attn_attr_bar_emb = self.image_embedder_attr(img_attn_attr_bar[:, 0, :])
        img_attn_emb = self.image_embedder(img_attn[:, 0, :])

        # Text embeddings
        concept = self.compose_word_embeddings(pairs_idx)

        # Classification logits
        logit_attr = self.cos_logits(img_attn_attr_emb, self.attr_embedder.weight)
        logit_attr_bar = self.cos_logits(img_attn_attr_bar_emb, self.attr_embedder.weight)
        logit_obj = self.cos_logits(img_attn_obj_emb, self.obj_embedder.weight)
        logit_obj_bar = self.cos_logits(img_attn_obj_bar_emb, self.obj_embedder.weight)
        logit_comp = self.cos_logits(img_attn_emb, concept)

        # Classification losses
        loss_comp = F.cross_entropy(self.scale * logit_comp, pairs)
        loss_obj = F.cross_entropy(self.scale * logit_obj, objs)
        loss_attr = F.cross_entropy(self.scale * logit_attr, attrs)
        loss_obj_bar = F.cross_entropy(self.scale * logit_obj_bar, objs)
        loss_attr_bar = F.cross_entropy(self.scale * logit_attr_bar, attrs)

        # Mask losses for samples without valid same_prim pairs
        # When mask=0, the sampled image is random, so we should reduce its contribution
        total_loss = loss_comp + \
                     (loss_obj + loss_obj_bar) * same_obj_mask.mean() + \
                     (loss_attr + loss_attr_bar) * same_attr_mask.mean() + \
                     loss_emd

        return (total_loss, score_obj, score_attr, score_dis)

    def _val_forward(self, batch, pairs_idx):
        img = batch[0].cuda()

        img_feat = self.extract_features(img)

        img_attn = self.cross_attn(q=img_feat, k=img_feat)
        img_attn_attr, _ = self.cross_attn_attr(q=img_feat, k=img_feat, return_attention=True)
        img_attn_obj, _ = self.cross_attn_obj(q=img_feat, k=img_feat, return_attention=True)

        # Compose text embeddings using all pairs
        concept = self.compose_word_embeddings(pairs_idx)

        img_proj = self.image_embedder(img_attn[:, 0, :])
        img_proj_obj = self.image_embedder_obj(img_attn_obj[:, 0, :])
        img_proj_attr = self.image_embedder_attr(img_attn_attr[:, 0, :])

        pair_pred = self.cos_logits(img_proj, concept)
        obj_pred = self.cos_logits(img_proj_obj, self.obj_embedder.weight)
        attr_pred = self.cos_logits(img_proj_attr, self.attr_embedder.weight)

        # Return as (comp_logits, attr_logits, obj_logits) for logit_infer
        return (pair_pred, attr_pred, obj_pred)

    def loss_calu(self, predict, batch):
        if self.training:
            # predict = (total_loss, score_obj, score_attr, score_dis)
            return predict[0]
        else:
            # Eval mode: compute simple CE loss for monitoring
            pair_pred, attr_pred, obj_pred = predict
            attrs = batch[1].cuda()
            objs = batch[2].cuda()
            pairs = batch[3].cuda()
            loss = F.cross_entropy(self.scale * pair_pred, pairs) + \
                   F.cross_entropy(self.scale * attr_pred, attrs) + \
                   F.cross_entropy(self.scale * obj_pred, objs)
            return loss

    def logit_infer(self, predict, pairs_idx):
        """Convert predictions to [batch, num_pairs] logits for evaluation."""
        pair_pred, attr_pred, obj_pred = predict

        # Combine: pair_score + aow * (attr_score * obj_score)
        # pairs_idx: [num_pairs, 2] with (attr_idx, obj_idx)
        attr_scores = attr_pred[:, pairs_idx[:, 0]]  # [batch, num_pairs]
        obj_scores = obj_pred[:, pairs_idx[:, 1]]    # [batch, num_pairs]

        logits = pair_pred + self.aow * attr_scores * obj_scores
        return logits
