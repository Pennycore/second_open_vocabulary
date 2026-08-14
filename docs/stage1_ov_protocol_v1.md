# Stage 1 OV-WSSS protocol v1 — OpenAI CLIP architecture

This supersedes the unexecuted v0 design. The forward encoder is **OpenAI CLIP
ViT-B/32 quick-GELU**, loaded strictly from the registered 512-dimensional
checkpoint. RemoteCLIP remains a historical baseline only.

The first-paper RemoteCLIP region features and visual prototypes must not be
scored with OpenAI CLIP text features. Any LoveDA Stage 1 run therefore needs
OpenAI CLIP image features re-encoded from the existing, read-only candidate
masks and RGB pixels. This is not permission to rerun SAM3, generate
pseudo-labels, or train a student.

Phase 1 remains a no-training held-out diagnostic. It is blocked until the
immutable OpenAI-CLIP pixel package, row-key/source hashes, tokenizer/prompt
manifest, and pre-result analysis commit are registered. Pixel GT and LoveDA
Val remain unavailable before the final method is frozen.
