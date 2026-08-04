# Versioning

Model versions and dataset versions are separate counters. A model version always names the
dataset version it was trained on; a dataset version never refers to a model version.

```
model v1  = stock Qwen2.5-1.5B-Instruct        (no dataset)
model v2  = LoRA over v1, trained on           dataset v1
model v3  = LoRA over v1, trained on           dataset v2      (more data)
model v4  = LoRA over v1, trained on           dataset v2      (different hyperparameters)
```

Note v3 and v4: a new model version does **not** require a new dataset. Changing rank, epochs
or learning rate produces a new model version on the same data. That is the point of keeping
the counters separate — you can tell at a glance whether a change came from the data or from
the training run.

## When to bump the dataset version

Bump when the output distribution changes. Specifically:

- the generator prompt changes (personas, complications, rules given to the generator)
- a quality rule is added, removed or loosened
- the generator model changes (`llama3.1:8b` → something else)
- temperature or other sampling settings change
- a new full run is generated, even with identical settings

That last one is deliberate. Two runs with the same settings are still different data, and
calling them both "v1" makes a training result impossible to attribute.

Do **not** bump for: adding rows to an existing run via `--resume`, or changing the business
name and context. The business is a runtime parameter recorded in the manifest, not a
property of the dataset design.

## When to bump the model version

Bump for any change that produces a different adapter:

- trained on a different dataset version
- any hyperparameter change (rank, alpha, learning rate, epochs, target modules)
- a different base model
- a different loss-masking or splitting strategy

Keep the base model in the version table. If you ever move off Qwen2.5-1.5B-Instruct, the
comparison against v1 stops being meaningful and you need a new baseline.

## What gets recorded

**Datasets** — the manifest is written automatically next to every generated file:

| Field | Why it matters |
|---|---|
| `dataset_version` | The label you pass with `--dataset-version` |
| `generator.model`, `.temperature`, `.seed` | Reproduce the sampling |
| `generator.quality_gate` | Whether defects were filtered — a big distribution difference |
| `code.generate_dataset.py`, `code.check_quality.py` | SHA-256 of the code that ran. Catches silent rule changes. |
| `assistant_system_prompt` | The prompt baked into every example |
| `scenario_counts` | Persona/complication/intent/service distribution |
| `data_sha256_16` | Identifies the exact file |

**Models** — `train_lora.py` writes `hyperparameters.json` into every checkpoint directory.
Add the dataset version to the row in [VERSIONS.md](../VERSIONS.md) when you train.

## Evaluating a new version

Always compare against v1, not against the previous version only:

```bash
python3 training/infer.py --base-only                        # v1
python3 training/infer.py --adapter checkpoints/ethos-v2     # v2
```

Run the same set of calls through both. A version that improves on v2 but is worse than v1 on
some axis is worth knowing about, and you will not see it comparing adjacent versions.
