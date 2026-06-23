"""Frozen CLIP backbone that exposes multi-scale patch features.

Uses vanilla open_clip and forward-hooks on the visual transformer blocks,
so it does NOT depend on any modified CLIP. The whole backbone is frozen.

NOTE (verify once): attribute names below match open_clip's ViT
(visual.transformer.resblocks, visual.ln_post, visual.proj,
 token_embedding, positional_embedding, transformer, ln_final, text_projection).
If your open_clip version differs, adjust the few flagged lines.
"""
import torch
import torch.nn as nn
import open_clip


class CLIPBackbone(nn.Module):
    def __init__(self, name="ViT-L-14-336", pretrained="openai",
                 feature_layers=(6, 12, 18, 24), img_size=518):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, force_image_size=img_size
        )
        self.clip = model.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        self.tokenizer = open_clip.get_tokenizer(name)
        self.feature_layers = list(feature_layers)
        self.width = self.clip.visual.proj.shape[0]      # e.g. 1024
        self.embed_dim = self.clip.visual.proj.shape[1]  # e.g. 768

        # register hooks on chosen resblocks (1-indexed -> 0-indexed)
        self._feats = {}
        blocks = self.clip.visual.transformer.resblocks
        for li in self.feature_layers:
            idx = li - 1
            blocks[idx].register_forward_hook(self._make_hook(li))

    def _make_hook(self, layer_id):
        def hook(module, inp, out):
            # open_clip Transformer: inp=(B, 1+N, width), out=(B, 1+N, width)
            self._feats[layer_id] = out
        return hook

    @torch.no_grad()
    def encode_image(self, x):
        """Returns (cls_embed (B,d), [patch_embed (B,N,d) per layer])."""
        self._feats = {}
        cls = self.clip.encode_image(x)  # triggers hooks, returns pooled cls
        cls = cls / cls.norm(dim=-1, keepdim=True)
        patch_feats = []
        for li in self.feature_layers:
            t = self._feats[li]              # (B, 1+N, width)
            t = t[:, 1:, :]                  # drop cls token -> patch tokens
            t = self.clip.visual.ln_post(t)  # FLAG: ln_post
            t = t @ self.clip.visual.proj    # FLAG: proj -> (B, N, embed_dim)
            t = t / t.norm(dim=-1, keepdim=True)
            patch_feats.append(t)
        return cls, patch_feats

    # ---- text encoding given pre-built token embeddings (for learnable prompts) ----
    def encode_text_from_embeddings(self, prompt_embeds, tokenized):
        """prompt_embeds: (n_prompts, L, width) already including ctx tokens.
        tokenized: (n_prompts, L) token ids (to locate EOT position)."""
        x = prompt_embeds + self.clip.positional_embedding 
        attn_mask = getattr(self.clip, "attn_mask", None)  # FLAG: transformer.attn_mask
        x = self.clip.transformer(x, attn_mask=attn_mask)
        #x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x)
        eot = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot] @ self.clip.text_projection
        return x / x.norm(dim=-1, keepdim=True)

    @property
    def token_embedding(self):
        return self.clip.token_embedding
