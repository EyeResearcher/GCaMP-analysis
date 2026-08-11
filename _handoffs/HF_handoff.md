# Hugging Face Model Migration Handoff

## Objective

Move the production ROI and spike classifiers out of the GCaMP-analysis GitHub
repository and into one versioned Hugging Face model repository. The application
must download and cache the correct release automatically so a non-coder does
not need the Hugging Face CLI, Git LFS, or manual file placement.

This migration is not complete if the files are merely uploaded. Loading,
validation, configuration, documentation, tests, and a friendly first-run and
offline experience are part of the work.

## Current State

### Production artifacts

The production bundle is the four tracked files in top-level `models/`:

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `models/roi_classifier.joblib` | Random Forest ROI classifier | `892003f2073840cb7958d85a1543574dd4738ff70de614d83b993b83e4296c04` |
| `models/roi_classifier_results.json` | ROI metrics and required `sqrt` transform | `8b896816894b7a58efa364d820a8a3a62641891fbf4c0b346ec7a9997339160b` |
| `models/spike_classifier.joblib` | Logistic Regression spike classifier | `8943bb8cde27869d6f2464e14b108dc4d3bfd1a29fdde249641603283df0e931` |
| `models/spike_classifier_results.json` | Spike metrics and required `raw` transform | `bcaf30d631eecbed27ab9c2b73ab37fa2177cfa1f19f4fc16d0153a6a2ed1a06` |

The JSON sidecars are runtime inputs, not optional reports. Loading the ROI
model without its sidecar silently omits the required `sqrt` transform and
changes inference behavior.

The ROI classifier expects these ordered features:

```text
peak_density
range_trace
derivative_skew
ac_decay
var_of_var
```

The spike classifier expects these ordered features:

```text
spike_prom
mini_prom
distance
```

Both production estimators were serialized with scikit-learn 1.6.1.

### Legacy artifacts

Do not publish `spike_classifier/models/` as part of the production release.
Those files are from older spike-classifier workflows:

- `spike_classifier.joblib` lacks `feature_names_in_` and is rejected by the
  current `utils.inference.get_model_feature_names` implementation.
- `spike_classifier.pkl` and `spike_classifier_all_features.pkl` are wrapper
  dictionaries using obsolete features such as `spike_prob_value`,
  `skew_contribution`, and `max_second_derivative_raw`.
- The feature, scaling, and transformation files belong to those legacy models.

If historical preservation is desired, put them in a clearly labeled archive
release or separate archive repository. The application must not discover them
as current models.

### Current loading flow

1. `main.py` loads the pipeline YAML.
2. It calls `utils.io_utils.load_model(config["models"], which="roi")` and
   again with `which="spike"`.
3. `load_model` resolves only local paths, calls `joblib.load`, and loads an
   optional JSON sidecar.
4. Both model/config pairs pass through `VideoPipelineRunner` into `ROIService`
   and `SpikeService`.
5. `utils.inference.prepare_features` uses `feature_names_in_`, orders the
   extracted columns, and applies the transform from the JSON sidecar.

Related readiness problems to fix in this migration:

- `config/notebook_config.yaml` contains absolute paths from one Windows host.
- `main.py` and the README refer to `config/pipeline_config.yaml`, but it is
  absent.
- `config/` and `models/` are broadly ignored in `.gitignore`, although some
  files under them are already tracked.
- `requirements.txt` and `environment.yml` allow `scikit-learn>=1.4`; loading
  the 1.6.1 artifacts under 1.4.2 emits incompatibility warnings.
- `huggingface_hub` is not a project dependency.

## Recommended Hugging Face Repository

Use one model repository for the compatible ROI/spike pair. They form one
analysis release, and one tagged snapshot prevents accidental combinations of
incompatible classifiers.

Owner decisions still required:

- Hugging Face namespace: **TBD**
- Repository name: recommended `gcamp-analysis-models`
- Visibility: **TBD** (`public` is simplest for non-coders; `private` requires
  authentication and token setup)
- License and data-sharing language: **TBD**

Recommended layout:

```text
README.md
manifest.json
roi/
  model.joblib
  config.json
spike/
  model.joblib
  config.json
```

Rename the results sidecars to `config.json` within their classifier folders,
or otherwise give them unambiguous names. Preserve all metric and transform
fields.

`manifest.json` should be machine-readable and include at least:

```json
{
  "bundle_version": "1.0.0",
  "gcamp_analysis_compatibility": ">=TBD",
  "python_version": "TBD",
  "scikit_learn_version": "1.6.1",
  "joblib_version": "TBD",
  "models": {
    "roi": {
      "model": "roi/model.joblib",
      "config": "roi/config.json",
      "sha256": "892003f2073840cb7958d85a1543574dd4738ff70de614d83b993b83e4296c04",
      "features": [
        "peak_density",
        "range_trace",
        "derivative_skew",
        "ac_decay",
        "var_of_var"
      ],
      "transform": "sqrt"
    },
    "spike": {
      "model": "spike/model.joblib",
      "config": "spike/config.json",
      "sha256": "8943bb8cde27869d6f2464e14b108dc4d3bfd1a29fdde249641603283df0e931",
      "features": ["spike_prom", "mini_prom", "distance"],
      "transform": "raw"
    }
  }
}
```

If files are renamed during upload, their byte contents and hashes stay the
same. If estimators are reserialized, recompute hashes and rerun inference
regression tests before release.

Create a semantic tag such as `v1.0.0`. Configuration must pin a tag or,
preferably for maximum reproducibility, a full commit hash. Never load
production models implicitly from the mutable `main` branch.

## Model Card Requirements

The Hugging Face `README.md` should state:

- these are scikit-learn classifiers for GCaMP-analysis, not Transformers;
- the purpose of the ROI and spike stages;
- exact expected features and transforms;
- training-data provenance to the extent permitted by study/data policy;
- sensor type, acquisition rate/range, preprocessing, labeling protocol, and
  biological conditions represented in the training data;
- evaluation metrics already present in the JSON sidecars;
- intended use and known limitations/generalization boundaries;
- exact Python, NumPy, SciPy, pandas, joblib, and scikit-learn versions used;
- the compatible GCaMP-analysis release or commit;
- a warning to load joblib/pickle artifacts only from a trusted source;
- a minimal usage example pointing users to GCaMP-analysis instead of manual
  downloads.

Do not claim validation beyond the existing evaluation records.

## Application Changes

### Dependencies and compatibility

Add `huggingface_hub` to `requirements.txt` and `environment.yml`.

Pin scikit-learn to the serialization version used by the first release:

```text
scikit-learn==1.6.1
```

If a supported range is desired later, validate predictions under every
supported version before widening it. Do not suppress compatibility warnings
as a substitute for validation.

### Configuration contract

Support Hub configuration similar to:

```yaml
models:
  source: "huggingface"
  repo_id: "<namespace>/gcamp-analysis-models"
  revision: "<full-commit-hash-or-v1.0.0>"
```

Retain backward-compatible local overrides for developers and air-gapped use:

```yaml
models:
  source: "local"
  roi_model_path: "/path/to/roi/model.joblib"
  roi_config_path: "/path/to/roi/config.json"
  spike_model_path: "/path/to/spike/model.joblib"
  spike_config_path: "/path/to/spike/config.json"
```

Ship a portable `config/pipeline_config.yaml` selecting the pinned public Hub
bundle by default. Remove workstation-specific paths from the normal workflow.

### Download and resolution behavior

Implement a small model-bundle resolver, keeping download and deserialization
separate:

1. For `source: local`, validate and load the four configured local files.
2. For `source: huggingface`, call
   `huggingface_hub.snapshot_download(repo_id=..., revision=...)` with allow-list
   patterns for `manifest.json`, `roi/*`, and `spike/*`.
3. Use the standard Hugging Face cache; do not download into the Git checkout.
4. Validate the manifest and SHA-256 hashes before `joblib.load`.
5. Validate scikit-learn version, expected feature names, and transforms against
   the manifest and sidecars.
6. Return the same `(model, config)` values currently expected by `main.py`.
7. Resolve/download once per process and bundle, not once per model or video.

On first run, tell the user the models are downloading and will be cached.
Cached subsequent runs should be quiet.

If the first run has no network, raise a concise error explaining that the
pinned bundle is not cached and show the local override option. A complete
cached snapshot must continue to work offline.

Do not require the `hf` CLI for application use. It is only for the maintainer's
upload/release workflow.

### Validation and safety

Joblib is pickle-based and can execute code during deserialization. Therefore:

- pin the repository revision;
- verify expected checksums before deserialization;
- load only from the configured trusted repository;
- never accept a model repository supplied through experiment data;
- never silently fall back to another revision;
- treat missing or malformed sidecars as hard failures;
- show expected and actual values for version/hash/feature mismatches.

Converting to `skops` may be evaluated separately, but do not mix it into this
migration unless prediction equivalence and dependency impact are tested.

## Tests Required

Add unit tests for:

- existing local-path loading;
- Hub resolution with `snapshot_download` mocked (no test network access);
- one bundle download for both classifiers;
- exact configured repository and pinned revision usage;
- checksum success and mismatch failure;
- missing/malformed manifest failure;
- missing model or config sidecar failure;
- scikit-learn version mismatch with a useful error;
- feature-name and transform mismatches;
- cached/offline success and uncached/offline failure;
- rejection of legacy artifacts in `spike_classifier/models/`.

Add a regression/smoke test with representative fixed feature rows for each
production model. Record expected predictions before removing the GitHub
copies, then assert identical predictions from the downloaded release. Ideally
record `predict_proba` values with a reasonable numeric tolerance as well.

Run the full test suite after integration.

## Migration Sequence

1. Confirm namespace, repository name, visibility, and license.
2. Capture training/runtime dependency versions and fill all `TBD` manifest and
   model-card fields.
3. Create the Hugging Face repository and upload only the production bundle,
   manifest, and model card.
4. Tag the verified release and record its full commit hash.
5. Implement dependencies, resolver, validation, portable configuration, and
   documentation in GCaMP-analysis.
6. Add and pass model-loading and prediction-regression tests.
7. Test a clean first run, cached second run, and offline cached run on a path
   different from the original Windows workstation.
8. Only after the pinned release works, remove the four production artifacts
   from the current GitHub tree.
9. Archive or remove `spike_classifier/models/`, clearly marking it unsupported
   by current inference.
10. Update training docs so new models go to a user output directory and
    publishing is an explicit maintainer workflow.

Deleting the files in a new commit does not erase historical blobs. A history
rewrite is not recommended solely for these artifacts: together they are only
about 1.1 MB while the repository pack is much larger. Treat history rewriting
as a separate, explicitly approved maintenance task.

## Acceptance Criteria

- A clean checkout contains no production `.joblib` or `.pkl` artifacts.
- `python main.py <experiment_root>` obtains the pinned compatible bundle
  without manual model setup.
- Models are cached and not repeatedly downloaded.
- Cached models work offline.
- The ROI sidecar and `sqrt` transform cannot be omitted silently.
- Unverified, incompatible, or malformed artifacts fail before inference and,
  where possible, before deserialization.
- Predictions match pre-migration regression fixtures.
- Local overrides continue to work.
- No application workflow requires the HF CLI or a token for a public repo.
- Documentation identifies the release, limitations, dependency versions, and
  recovery steps clearly enough for a non-coder.

## Out of Scope Unless Separately Requested

- Retraining either classifier.
- Changing algorithms, thresholds, features, or scientific behavior.
- Publishing training or labeled experimental data.
- Rewriting Git history.
- Deploying a Hugging Face Inference Endpoint or Space; these small classifiers
  should run locally after download.
