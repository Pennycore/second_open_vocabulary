# Research traceability archive

This directory is the durable, repository-tracked index for the second-paper experiments. It is deliberately separated into:

- tracked metadata: chronology, decisions, experiment registry, material passport, curated inventory, source-commit map, and checksums for dated payloads;
- ignored `artifacts/`: immutable, dated local and server snapshots, because they are large and may contain local-only material;
- ignored `private/`: exact machine-path inventory and post-commit bundle checksum. It is never committed.

## What is preserved

The two dated tar archives contain the experiment outputs, reports, configurations, documentation, and source snapshot present on their respective machines. The server snapshot includes the server workspace's uncommitted experimental code as well as its outputs; the local snapshot includes the local source tree and the pre-existing transient Vaihingen helper scripts. Raw datasets, inputs, and weights are not duplicated; manifests and source documents record their identifiers and hashes where available.

`repository_full_20260820.bundle` is the canonical historical source archive. It is a full Git `--all` bundle made after this metadata index was committed. To restore a source tree, clone it into a new directory and select a commit:

```text
git clone repository_full_20260820.bundle restored_open_vocabulary
cd restored_open_vocabulary
git log --all --oneline
git switch --detach <recorded-commit>
```

The bundle preserves committed source history and refs. Uncommitted or transient source is preserved only in the dated source snapshots; it must not be attributed to a Git commit. The bundle checksum is retained in the ignored private inventory to avoid a self-referential checksum cycle.

## Scope and limits

This is a provenance aid, not a claim of deterministic replay. The conversation transcript was not fully exported. LLM reasoning is not byte-reproducible, and the exact Codex runtime model identifier was not captured by the project. See [material_passport.json](material_passport.json), [research_timeline.md](research_timeline.md), and [decision_log.md](decision_log.md).
