# Architecture decision v1 — OpenAI CLIP

**Decision:** all forward work in the second-paper pipeline uses OpenAI CLIP
ViT-B/32 quick-GELU as its frozen visual-language encoder.

The completed VOC whole-image, no-training sanity check favored OpenAI CLIP on
that fixed natural-image diagnostic. This decision is an architecture choice,
not a claim that it has already improved LoveDA region segmentation.

RemoteCLIP Stage 0 outputs remain immutable historical evidence. They may be
used to reproduce their original RemoteCLIP-space analysis, but their feature
vectors, text vectors, and prototypes cannot be mixed with OpenAI CLIP. Future
LoveDA work must re-encode the frozen candidate RGB/mask views in OpenAI CLIP
space; it must not rerun SAM3 or start student training under this decision.
