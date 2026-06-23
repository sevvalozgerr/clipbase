"""Object-agnostic learnable prompts (CoOp-style), AnomalyCLIP formulation.

Normal : [V_1]...[V_n] object
Anomaly: [W_1]...[W_n] damaged object

Learns two sets of context vectors. Produces 2 text embeddings: (normal, anomaly).
"""
import torch
import torch.nn as nn
import open_clip


class ObjectAgnosticPrompts(nn.Module):
    def __init__(self, backbone, n_ctx=12):
        super().__init__()
        self.backbone = backbone
        width = backbone.token_embedding.weight.shape[1]
        self.n_ctx = n_ctx

        # learnable context for normal and anomaly streams
        self.ctx_normal = nn.Parameter(torch.empty(n_ctx, width))
        self.ctx_anomaly = nn.Parameter(torch.empty(n_ctx, width))
        nn.init.normal_(self.ctx_normal, std=0.02)
        nn.init.normal_(self.ctx_anomaly, std=0.02)

        tokenizer = backbone.tokenizer
        # suffix tokens (fixed): "object." and "damaged object."
        self.register_buffer("tok_normal", tokenizer(["object."])[0])
        self.register_buffer("tok_anomaly", tokenizer(["damaged object."])[0])

        with torch.no_grad():
            emb_n = backbone.token_embedding(self.tok_normal.unsqueeze(0))[0]
            emb_a = backbone.token_embedding(self.tok_anomaly.unsqueeze(0))[0]
        # SOS + suffix tokens (we keep SOS at pos0, then insert ctx, then suffix words)
        self.register_buffer("emb_normal", emb_n)   # (L, width)
        self.register_buffer("emb_anomaly", emb_a)

    def _assemble(self, ctx, suffix_emb, suffix_tok):
        # suffix_tok layout: [SOS, w1, w2, ..., EOT, pad, ...]
        ctx_len = suffix_emb.shape[0] # max length of context + suffix
        #L = suffix_emb.shape[0]
        sos = suffix_emb[:1]                       # (1, width)
        rest = suffix_emb[1:]                      # words + EOT + pad
        full = torch.cat([sos, ctx, rest], dim=0)[:ctx_len]  # (1+n_ctx+rest, width)
        #full = full[:L]                            # keep CLIP context length
        # rebuild a matching token-id row so EOT position is correct
        tok = suffix_tok
        # shift suffix ids right by n_ctx (ctx positions hold no real id; use 0)
        new_tok = torch.zeros_like(tok)
        new_tok[0] = tok[0]
        words = tok[1:ctx_len - self.n_ctx]
        new_tok[1 + self.n_ctx: 1 + self.n_ctx + words.shape[0]] = words
        return full.unsqueeze(0), new_tok.unsqueeze(0)

    def forward(self):
        en, tn = self._assemble(self.ctx_normal, self.emb_normal, self.tok_normal)
        ea, ta = self._assemble(self.ctx_anomaly, self.emb_anomaly, self.tok_anomaly)
        embeds = torch.cat([en, ea], dim=0)   # (2, L, width)
        toks = torch.cat([tn, ta], dim=0)     # (2, L)
        text = self.backbone.encode_text_from_embeddings(embeds, toks)  # (2, d)
        return text  # row0 = normal, row1 = anomaly
