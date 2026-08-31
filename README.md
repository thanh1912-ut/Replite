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
