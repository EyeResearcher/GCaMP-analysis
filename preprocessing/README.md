# `preprocessing`

Optional helpers for producing the Suite2p output that the analysis pipeline
consumes. These are batch-automation wrappers around Suite2p, not part of the
`gcamp_analysis` library, and they are not required if you already have
`suite2p/plane0/` folders.

## Modules

| Module | Responsibility |
|---|---|
| `batch_s2p.py` | Recursively run Suite2p over a tree of TIFFs, reusing your saved Suite2p GUI ops so batch runs match interactive settings. |
| `suite2p_compat.py` | Compatibility shims that smooth over differences between Suite2p versions/return signatures. |

## Usage

```bash
python preprocessing/batch_s2p.py --fs 15 --pretrained_model invitro_rgcs_max
```

## Notes

- `batch_s2p.py` contains machine-specific paths (Suite2p ops location and a
  temp output directory) near the top of the file. Update these to your own
  environment before running.
- The pipeline only requires the resulting `suite2p/plane0/` arrays; how they
  are generated is up to you.
