# RepLite

RepLite is a lightweight, statically configurable multi-task perception model.
It combines one of two ImageNet-1K backbones with always-on LiteConvLSTM
feature refinement, a task-prunable neck, and portable heads for detection,
segmentation, depth, and classification.

The package also keeps the standalone backbone API described below.

## Complete multi-task model

```python
import torch

from replite import RepLiteConfig, TaskConfig, create_replite_model

config = RepLiteConfig(
    backbone_name="mobilenetv4_conv_small",
    pretrained=True,
    tasks=TaskConfig(
        detection_classes=10,
        segmentation_classes=3,
        depth=True,
    ),
    recurrence_steps=3,
)
model = create_replite_model(config).eval()

images = torch.randn(2, 3, 384, 640)
output = model(images)
print(output.segmentation.shape)  # (2, 3, 384, 640)
print(output.depth.shape)         # (2, 1, 384, 640), strictly positive
print(len(output.detection.cls_logits))  # P3, P4, P5
```

Inputs are RGB NCHW tensors normalized by the caller. Class counts are always
explicit; RepLite does not assume a dataset label schema. A lane or
drivable-area task uses the segmentation head with the appropriate class
count and application loss.

### Recurrent behavior

LiteConvLSTM is enabled for every configuration:

- A 4D image runs the backbone once, then repeats the task-required C4 and/or
  C5 feature through the recurrent cells for `recurrence_steps`
  iterative-refinement steps.
- A 5D `B,T,3,H,W` clip runs in sequence mode and returns the final-frame
  prediction. Recurrent state advances on every frame, while task decoders and
  heads run only for the final frame.
- Streaming inference uses explicit state; no state is hidden inside the
  module, so video/session boundaries remain under caller control.

```python
state = None
for frame in stream:
    output, state = model.forward_step(frame, state)

# At a truncated-BPTT boundary:
from replite.multitask import detach_state
state = detach_state(state)

# At a new video/session:
state = None
```

For static data, `model(image)` and `model.forward_static(image)` use recurrent
refinement by default. Do not describe this as motion modeling; it is
iterative feature refinement. Real clips use spatiotemporal aggregation.

In training mode, the backbone processes a clip as one flattened `B*T` batch
for efficient BatchNorm statistics. Consequently, splitting one clip into
different truncated-BPTT chunk sizes is not numerically equivalent while
BatchNorm is updating. Use fixed chunking, or freeze/evaluate BatchNorm before
expecting chunked continuation parity.

### Task-dependent artifacts

Only configured paths and heads are materialized:

```python
detection_model = create_replite_model(
    RepLiteConfig(tasks=TaskConfig(detection_classes=8))
)
depth_model = create_replite_model(
    RepLiteConfig(tasks=TaskConfig(depth=True))
)
```

A detection-only model contains no dense decoder. A dense-only model contains
no LitePAN and physically stops the backbone at C4. A classification-only model
keeps recurrent C5 but removes recurrent C4 and both prediction paths.
Segmentation-depth gated fusion exists only when both heads are enabled. These
static artifacts have no disconnected trainable parameters under their own
complete task loss, so normal DDP does not require unused-parameter discovery.

For mixed datasets with partial labels, compute a loss only when that target is
present. If an entire DDP iteration omits one enabled head, use
`find_unused_parameters=True` or construct task-balanced iterations. Keep
`gated_dense_fusion=False` unless segmentation and depth are jointly supervised;
otherwise build separate adapters without cross-task coupling.

Task wrappers return tensors only and are the supported static-image export
surface. A wrapper routes only through the selected task's dependencies; when
segmentation-depth fusion is trained, it retains both adapters and the fusion
needed to preserve that task's learned semantics. `export_task()` takes an
independent snapshot of the current weights, so create it after loading the
final checkpoint (and recreate it after any later weight update):

```python
model.eval().switch_to_deploy()
segmentation_graph = model.export_task("segmentation")
exported = torch.export.export(
    segmentation_graph,
    (torch.randn(1, 3, 384, 640),),
)
print(segmentation_graph.model_metadata)
```

The detection wrapper returns nine tensors in this order: three class-logit
maps, three box-regression maps, and three quality-logit maps for P3/P4/P5.
Assignment, task losses, box decoding, and NMS are intentionally outside the
model because they depend on the dataset/training stack. Save
`model.model_metadata` beside training checkpoints and the wrapper's
`model_metadata` beside each task-specific export.

## Training contract

`replite.training` supplies a reference training stack without changing the
inference model API. A loader yields either `(inputs, targets)` or a mapping
with `inputs` (or `images`) and `targets`. `inputs` can be a static
`B,3,H,W` tensor or a clip `B,T,3,H,W`; every target describes the final
frame predicted by the model.

The target mapping uses explicit task keys:

```python
targets = {
    "detection": [
        {
            "boxes": boxes_xyxy,      # float N,4; absolute half-open pixels
            "labels": labels,         # int64 N; zero-based class IDs
            "valid_size": (height, width),
            "ignore_boxes": crowd_xyxy,  # optional
        }
        for _ in range(batch_size)
    ],
    "segmentation": semantic_ids,      # int64 B,H,W; 255 is ignored
    "depth": depth_metres,             # float B,1,H,W
    "depth_valid": valid_depth_mask,   # bool B,1,H,W
    "classification": image_labels,    # optional; -100 is ignored
}
```

An empty `boxes` tensor is a labelled negative image. It is not equivalent to
missing detection supervision. Dense targets use their ignore mask/value for
partial labels, so an enabled task is never silently treated as a negative.

The reference detector uses FCOS points at strides 8, 16, and 32, deterministic
smallest-box assignment, focal classification, GIoU regression, centerness
quality, and optional DFL. Decoding combines class and quality probabilities,
then applies pure-PyTorch class-aware NMS. This keeps training and deployment
usable without adding a `torchvision` runtime dependency.

Validation accumulators report COCO-style detection mAP50--95, segmentation
mIoU/pixel accuracy, and standard metric-depth errors. Checkpoints contain the
model, optimizer, scheduler, AMP scaler, RNG state, progress, model metadata,
and caller-provided run metadata. Resume is exact at saved epoch/optimizer
boundaries; exact mid-epoch continuation additionally requires a resumable
sampler supplied by the application.

A minimal run is assembled explicitly:

```python
import math

from replite.data import SANPO_DETECTION_CLASS_NAMES
from replite.training import (
    CheckpointManager,
    DepthMetrics,
    DetectionMAP,
    MultiTaskCriterion,
    MultiTaskMetrics,
    SegmentationMetrics,
    Trainer,
    TrainerConfig,
    TrainingLogger,
    WarmupCosineScheduler,
    create_adamw,
)

num_detection_classes = len(SANPO_DETECTION_CLASS_NAMES)
criterion = MultiTaskCriterion(
    detection_num_classes=num_detection_classes,
    detection_reg_max=config.detection_reg_max,
    depth_loss_type="log_l1_silog",
    depth_min=0.1,
    depth_max=80.0,
)
optimizer = create_adamw(model, lr=3e-4, weight_decay=1e-2)
runtime = TrainerConfig(
    epochs=100,
    grad_accum_steps=2,
    amp=True,
    monitor="val/detection/map50_95",
    monitor_mode="max",
)
steps = runtime.epochs * math.ceil(len(train_loader) / runtime.grad_accum_steps)
scheduler = WarmupCosineScheduler(optimizer, total_steps=steps, warmup_steps=500)
metrics = MultiTaskMetrics(
    detection=DetectionMAP(num_detection_classes),
    segmentation=SegmentationMetrics(3),
    depth=DepthMetrics(min_depth=0.1, max_depth=80.0),
    detection_reg_max=config.detection_reg_max,
)
trainer = Trainer(
    model,
    criterion,
    optimizer,
    runtime,
    scheduler=scheduler,
    validation_metrics=metrics,
    logger=TrainingLogger("runs/sanpo_joint", run_id="sanpo_joint_seed42"),
    checkpoint_manager=CheckpointManager("runs/sanpo_joint/checkpoints"),
)
trainer.fit(train_loader, val_loader)
```

`Trainer.resume()` verifies SHA-256, then falls back from a corrupt `last.pt`
to `last.prev.pt`. It never hides a dataset/config/model metadata mismatch.
The application still owns synchronized clip transforms and DataLoader policy.

## SANPO-Real pilot and derived detection targets

Open
[`notebooks/SANPO_Real_Pilot_2train_1test_Detection.ipynb`](notebooks/SANPO_Real_Pilot_2train_1test_Detection.ipynb)
in Colab for the small data pilot. Its single executable cell:

- reuses the Drive discovery/inventory cache;
- selects two official-train sessions and one official-test session, with at
  most one camera per session;
- downloads only `RGB(t-2,t-1,t)`, `panoptic(t)`, and `depth(t)` joint samples;
- derives per-frame detection JSON plus a JSONL index before archiving; and
- publishes a checksum-verified, versioned archive without overwriting an old
  preprocessing configuration.

The official test session is for final evaluation only, never early stopping.
Derived boxes are not an official SANPO detection benchmark. SANPO stores the
semantic ID in RGB channel 0 and a 16-bit instance ID in channels 1--2. The
converter keeps `instance_id=0`, splits disconnected regions with
8-connectivity, and emits half-open XYXY boxes. By default components of at
least 100 source pixels are positives; smaller thing components become
`ignore_boxes`. This threshold is versioned because it comes from the paper's
object-counting convention rather than an official detection protocol.

The same pure conversion is available without Colab:

```python
from replite.data import load_sanpo_detection

target = load_sanpo_detection("segmentation_masks/000123.png", min_area=100)
# target: boxes float32 N,4; labels int64 N; valid_size; ignore_boxes
```

After the pilot archives exist, run
[`notebooks/SANPO_Real_Extract_Visualize_Smoke_Train.ipynb`](notebooks/SANPO_Real_Extract_Visualize_Smoke_Train.ipynb)
in a GPU Colab. It verifies every archive checksum and tar member, extracts to
local SSD, renders six provenance-backed RGB/box/semantic/depth QA panels, and
runs a session-disjoint 2--5 epoch smoke test (three by default). One
official-train session is train, the other is validation, and the official-test
session is limited to a final forward-only pipeline check. Logs, resolved
configuration, preview, atomic checkpoints, and a SHA-256 artifact manifest are
mirrored to a new Drive run directory only after the smoke gates pass.

The checked-in manifest adapter can also be assembled directly:

```python
from torch.utils.data import DataLoader
from replite.data import SanpoJointDataset, sanpo_joint_collate

dataset = SanpoJointDataset(
    "_sanpo_joint_manifest.json",
    image_size=(288, 512),
    depth_min=0.1,
    depth_max=80.0,
)
loader = DataLoader(dataset, batch_size=2, collate_fn=sanpo_joint_collate)
clip, targets = next(iter(loader))
print(clip.shape)  # B,3,3,288,512
```

Semantic source ID zero is unlabeled and becomes ignore value 255; source IDs
1--30 map to 30 contiguous segmentation classes. The adapter decodes SANPO
depth as gzip little-endian float16 metres, preserves an explicit validity
mask, and scales both positive and ignored half-open boxes with the dense
targets. It reads sparse sample IDs from the manifest rather than scanning a
numeric frame range.

### SANPO smoke AMP gate

The notebook starts FP16 gradient scaling at 4096. The first pilot run showed
that PyTorch's default 65536 scale was four reductions above the stable range:
four early optimizer updates were skipped while `GradScaler` backed off to
4096, although all forward losses remained finite and train/validation loss
decreased. The smoke audit now logs every overflow and the scale that follows
it. A run with skips is accepted only as `smoke_pass_amp_recovered` when skips
are at most 5% of attempted updates, the final epoch has zero new skips, and
the final scale is finite and at least one. Any late or persistent overflow is
a failure, not a warning to suppress.

If the original three-epoch Colab pilot already stopped only at its obsolete
zero-skip assertion, keep that runtime alive and run:

```python
!git -C /content/Replite pull --ff-only
%run -i /content/Replite/tools/recover_sanpo_smoke_amp.py
```

Do not rerun setup Cell 1 or training Cell 7 first. The recovery verifies the
live weights against `last.pt`, compares cumulative skips in `last.prev.pt`,
performs only the held-out forward-pass QA, and checks every copied artifact
before publishing to Drive.

### SANPO smoke validation metrics

For a completed live pilot, pull the current evaluator and run it in a new
Colab cell. Do not rerun setup, extraction, training, or AMP recovery first:

```python
!git -C /content/Replite pull --ff-only
%run -i /content/Replite/tools/evaluate_sanpo_smoke_val.py
```

The evaluator creates a separate inference model, strict-loads the selected
`best.pt`, and consumes only the held-out validation session. It reports
detection mAP50 and mAP50--95, segmentation mIoU and all per-class IoUs, and
depth AbsRel, RMSE, and delta1. JSON, tidy CSV tables, the complete 30-by-30
segmentation confusion matrix, and a PNG/SVG dashboard are checksum-verified
before being published under the run's versioned `evaluations/` directory.
The live training model, optimizer, scheduler, scaler, RNG state, root artifact
manifest, and official-test split are left unchanged.

These are descriptive pilot metrics from one 73-frame held-out official-train
session at 288x512. Detection boxes are derived from panoptic components and
the internal AP accumulator is not the official SANPO or COCO evaluator, so do
not present this bundle as an official benchmark result.

## SANPO-Real main training

Use
[`notebooks/SANPO_Real_Main_Train.ipynb`](notebooks/SANPO_Real_Main_Train.ipynb)
after all 234 downloader archives are present on Drive. The notebook audits the
expected 186 official-train session-camera archives (14,718 joint targets) and
48 official-test archives (3,803 targets). The current limited-data campaign
uses seed-bound SHA-256 ordering over the complete pools to freeze exactly 20
official-train sessions for fit, one distinct official-train session for
validation, and one official-test session for holdout. The holdout is catalogued
for provenance but is never extracted or passed to training, validation,
early stopping, or checkpoint selection.

The Drive ledger may contain 237 entries because the three pilot sessions were
repackaged after versioned detection JSON was introduced. Catalog resolution
still produces exactly 234 source shards: it prefers those three packages and
uses the 231 source-keyed legacy archives for the remainder. Legacy shards do
not need to be downloaded or repacked again; their boxes are derived from the
already-loaded panoptic mask with the same locked 8-connected, 100-pixel,
half-open-XYXY policy. The catalog fingerprint records which representation
was used for every shard.

Run the notebook cells in order. Cell 3 prints the resolved architecture,
MobileNet feature stages, ImageNet-1K weight provenance, parameter counts,
optimizer groups, schedule, archive hashes, frozen split, and SSD capacity plan
before any model update. The notebook anchors Drive at
`/content/drive/MyDrive/nckh1m_data` and accepts the locked SANPO marker layout
either directly there or in `sanpo_real_v0_joint_human_only_rgb3`; it never
searches arbitrary Drive locations. Inspection validates the three compact
metadata manifests without issuing per-archive FUSE `resolve/stat` calls. Cell
4 then copies, byte/SHA-verifies, and atomically extracts only the shards
belonging to the seed-42 hash-selected 20 fit sessions plus one distinct
inner-validation session once to `/content`. A separate hash-selected
official-test session is frozen in the immutable split manifest but is not
opened by fit, validation, early stopping, or checkpoint selection.
Cell 5 performs a disposable one-batch preflight and then runs exactly campaign
epoch 1 over the complete train split with complete validation. If FP16 scaled
gradients overflow, preflight records the affected parameters and backs the
scale down deterministically before constructing the production trainer. Cell
6 displays the operational gate and approval token. Only after reviewing those
results should `START_MAIN` be enabled and the token pasted into Cell 7; that
cell strict-resumes epoch 2 with the same full-campaign scheduler.

The console uses newline-safe YOLO-style rows that remain readable both inside
the running cell and through `tail -F`. Before training it prints the exact
train/validation sample counts, micro/effective batch size, batches per epoch,
optimizer updates, warmup updates, workers, prefetch, and log cadence. During
SSD staging it reports ready/total shards, overall GiB and percentage, current
session-camera, copy/extract/verify phase, throughput, ETA, and remaining SSD
space. Train and validation rows include current/total batch, percentage,
iterations per second, ETA, task losses, VRAM, instances, resolution, and LR.
Wrapper keepalives appear only after 30 seconds with no child output, so they do
not obscure real progress. `console.log` contains only the current/latest
command for `tail -F`; immutable per-invocation logs live below `logs/`. Every
validation writes detection mAP50/mAP50--95 and per-class AP,
segmentation mIoU/pixel accuracy/per-class IoU, and depth AbsRel/RMSE/delta1.
Each completed epoch publishes a versioned SHA-256 snapshot containing the
checkpoint, history, metrics, resolved configuration, and split metadata.
After a Colab disconnect, rerun setup/config and Cell 4 because `/content` is
ephemeral, then read the existing gate, skip the pilot, and use Cell 7; restore
scans newest to oldest and rejects corrupt or source/config/catalog/split-
incompatible snapshots. Each `RUN_ID` also writes an immutable source pin on
Drive, so Cell 2 checks out the campaign's original commit even if `main` has
advanced since the previous Colab session.

The 20-fit/1-val/1-test campaign has its own source pin and private local stage
cache. (`/content` is ephemeral, so no SSD cache survives a runtime reset.)
`LocalArchiveStage` still verifies the dataset-owned stage marker, archive
identity, and referenced payload signatures before reuse.

The staged loader reuses one global map-style dataset and one persistent worker
pool across all selected shards, so batches can cross session boundaries and
there is only one tail batch per split. Cell 4 also builds an atomic, versioned
local cache of resized RGB, segmentation, lossless float16 depth, valid masks,
and exact detection boxes. Later epochs therefore do not reread Drive,
re-extract archives, decode PNG/depth gzip, or rerun connected components. Both
archive staging and cache warming are resumable and print progress, throughput,
ETA, and disk planning. Capacity planning uses the downloader's exact selected-
file byte totals plus a configurable allowance. After full
campaign completion, Cell 8 may reclaim that private train/val cache and stage
only the single frozen official-test session. It is blocked before full
campaign completion, and staging alone does not evaluate the holdout. Derived
detection boxes remain an internal panoptic-to-box protocol,
not an official SANPO detection benchmark.

This is explicitly a 20-session limited-data experiment. Its metrics must not
be reported as full-corpus SANPO results; the selected session IDs and policy
are stored in the immutable split manifest and bound into snapshot parity.

## Standalone backbones

Lightweight native-trunk feature backbones for dense prediction:

- MobileNetV3-Small ×0.5
- MobileNetV4-Conv-S

Both expose native C2–C5 features and can strict-load SHA-256-pinned
ImageNet-1K safetensors checkpoints.

```python
from replite.backbone import create_backbone

backbone = create_backbone(
    "mobilenetv4_conv_small",
    pretrained=True,
    out_indices=(0, 1, 2, 3),
    cache_dir="/content/drive/MyDrive/model_cache",
)
features = backbone(images)
print(backbone.feature_info.channels())
print(backbone.weights_provenance)
```

Inputs must be RGB NCHW tensors normalized by the caller. Save
`backbone.backbone_config` beside every training checkpoint so initialization
provenance remains explicit.

`out_indices` trims the registered trunk after the deepest requested stage so
all trainable parameters participate in backward/DDP. State dictionaries are
therefore directly portable between selections only when their deepest stage
is the same.

For an offline verified copy of the official checkpoint, pass
`checkpoint_path=...` together with `pretrained=True`. The file must match the
pinned SHA-256; arbitrary fine-tuned `.pt` checkpoints belong in the normal
training checkpoint loader instead.

## Tests

```bash
python -m pytest -q
python -m pytest -q -m network
python -m pytest -q -m cuda
```

The CUDA marker requires an NVIDIA runner and is intentionally invoked
explicitly; the checked-in GitHub workflows use public CPU runners only.

## Migrating from 0.1

Version 0.2 intentionally replaces `models.backbone` with
`replite.backbone`; keeping the old package as a shim would recreate the
collision with applications that already own a top-level `models` package.
Update imports as follows:

```python
# 0.1
from models.backbone import create_backbone

# 0.2+
from replite.backbone import create_backbone
```

Version 0.1 shallow-subset state dictionaries contained the complete trunk.
Load those safely into a trimmed 0.2 model with:

```python
ignored = backbone.load_legacy_full_trunk_state_dict(old_state_dict)
print("ignored legacy suffix keys:", ignored)
```

Whole-module pickle files reference the old Python module path. Open them in a
0.1 environment, export `state_dict()`, then use the migration method above;
do not add a `models` compatibility shim.
