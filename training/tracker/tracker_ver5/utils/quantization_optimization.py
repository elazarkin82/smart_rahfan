#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) Optimization Script
======================================================
This script applies QAT to a pre-trained Keras model to prepare it for
high-accuracy INT8 TFLite conversion. It supports:
  1. Selective QAT wrapping to bypass custom/subclassed layers.
  2. Teacher-Student Knowledge Distillation mode.
  3. Direct dataset label training mode.
  4. Per-output loss configuration via --output_ops (generic; driven by
     model.output_names at runtime — no hardcoded output name logic).
"""

import os
import sys
import argparse

import tensorflow as tf
# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# Run in graph mode (run_functions_eagerly=False) which is required to update MovingAverageQuantizer.
tf.config.run_functions_eagerly(False)

import tensorflow_model_optimization as tfmot

import numpy as np
import tqdm

# =====================================================================
# Loss Functions
# =====================================================================

def _soft_argmax_2d(heatmap, beta=30.0):
    """Differentiable soft-argmax over a (B, H, W, 1) heatmap.

    Returns a (B, 2) tensor of [y, x] coordinates in normalized [0, 1] units,
    computed as the softmax-weighted expectation over the spatial grid.
    """
    B = tf.shape(heatmap)[0]
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2]

    flat_hm = tf.reshape(heatmap, [B, H * W])
    probs = tf.nn.softmax(beta * flat_hm, axis=-1)  # (B, H*W)

    y_grid, x_grid = tf.meshgrid(
        tf.cast(tf.range(H), tf.float32) / tf.cast(H, tf.float32),
        tf.cast(tf.range(W), tf.float32) / tf.cast(W, tf.float32),
        indexing='ij',
    )
    flat_y = tf.reshape(y_grid, [-1])  # (H*W,)
    flat_x = tf.reshape(x_grid, [-1])  # (H*W,)

    pred_y = tf.reduce_sum(probs * flat_y, axis=-1, keepdims=True)  # (B, 1)
    pred_x = tf.reduce_sum(probs * flat_x, axis=-1, keepdims=True)  # (B, 1)
    return tf.concat([pred_y, pred_x], axis=-1)  # (B, 2)


def _gt_peak_coords(gt_heatmap):
    """Hard-argmax peak coordinates from a (B, H, W, 1) GT heatmap.

    Returns a (B, 2) tensor of [y, x] in normalized [0, 1] units.
    """
    H = tf.shape(gt_heatmap)[1]
    W = tf.shape(gt_heatmap)[2]
    flat_hm = tf.reshape(gt_heatmap, [-1, H * W])
    flat_idx = tf.argmax(flat_hm, axis=-1)
    W_64 = tf.cast(W, tf.int64)
    y = tf.cast(flat_idx // W_64, tf.float32) / tf.cast(H, tf.float32)
    x = tf.cast(flat_idx % W_64, tf.float32) / tf.cast(W, tf.float32)
    return tf.stack([y, x], axis=-1)  # (B, 2)


class SoftArgmaxCoordLoss(tf.keras.losses.Loss):
    """Soft-argmax coordinate Huber loss for heatmap regression.

    Given a predicted heatmap, extracts coordinates via a differentiable
    soft-argmax (temperature beta), then computes a per-sample Huber loss
    against the target coordinates (which are in [0, 1] relative range).

    In teacher-student (distillation) mode the target heatmap is the
    teacher's output.  A positive-sample mask is applied.

    Args:
        beta:            Soft-argmax temperature (higher → sharper, closer
                         to hard-argmax; default 30.0).
        huber_delta:     Delta for the Huber loss (default 1.0 pixel in 256px space).
        peak_threshold:  Minimum target heatmap peak value to consider a
                         sample positive (default 0.1).
        name:            Loss name shown in training logs.
    """

    def __init__(self, beta=30.0, huber_delta=1.0, peak_threshold=0.1,
                 name="soft_argmax_coord_loss"):
        super().__init__(name=name)
        self.beta = beta
        self.huber_delta = huber_delta
        self.peak_threshold = peak_threshold
        self._huber = tf.keras.losses.Huber(
            delta=huber_delta,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def call(self, y_true, y_pred):
        """Compute masked coordinate Huber loss.

        Args:
            y_true: target coordinates (B, 2) OR target heatmap (B, H, W, 1).
            y_pred: (B, H, W, 1) predicted heatmap from the student model.

        Returns:
            Scalar loss (mean over positive samples; 0 if no positives).
        """
        pred_coords_norm = _soft_argmax_2d(y_pred, beta=self.beta)   # (B, 2) in [0, 1]

        # Determine target format and mask dynamically
        if len(y_true.shape) == 2 or y_true.shape[-1] == 2:
            # y_true is ground_truth_coords (B, 2)
            gt_coords_norm = y_true
            # Positive sample: L2 norm of coordinates is non-zero (negatives are [0, 0])
            pos_mask = tf.cast(tf.reduce_sum(tf.square(y_true), axis=-1) > 1e-6, tf.float32)
        else:
            # y_true is teacher heatmap output (B, H, W, 1)
            gt_coords_norm = _gt_peak_coords(y_true)
            peak_val = tf.reduce_max(
                tf.reshape(y_true, [tf.shape(y_true)[0], -1]), axis=-1
            )  # (B,)
            pos_mask = tf.cast(peak_val > self.peak_threshold, tf.float32)

        pos_mask = tf.reshape(pos_mask, [-1, 1])                 # (B, 1)

        # Scale both coordinates to 256.0 space to compute Huber loss in pixel units
        coord_loss = self._huber(gt_coords_norm * 256.0, pred_coords_norm * 256.0) # (B,)
        coord_loss = tf.reshape(coord_loss, [-1, 1])             # (B, 1)

        masked_loss = coord_loss * pos_mask
        return tf.reduce_sum(masked_loss) / (tf.reduce_sum(pos_mask) + 1e-7)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta": self.beta,
            "huber_delta": self.huber_delta,
            "peak_threshold": self.peak_threshold,
        })
        return cfg


# =====================================================================
# Dataset helpers
# =====================================================================

class MemorySafeDataset:
    def __init__(self, h5_path, batch_size=16):
        import h5py
        import psutil

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset not found at {h5_path}")

        self.h5_path = h5_path
        self.batch_size = batch_size

        with h5py.File(h5_path, 'r') as f:
            self.total_samples = f['reference_stack'].shape[0]

        sample_size_bytes = (64 * 64 * 16 + 256 * 256 * 1) * 4 + 12
        total_expected_bytes = self.total_samples * sample_size_bytes

        mem = psutil.virtual_memory()

        if total_expected_bytes < mem.available * 0.35:
            print(f"[*] Pre-loading dataset into RAM ({self.total_samples} samples, "
                  f"{total_expected_bytes / (1024**3):.2f} GB)...")
            with h5py.File(h5_path, 'r') as f:
                self.ref_all    = f['reference_stack'][:]
                self.search_all = f['search_frame'][:]
                self.gt_coords_all  = f['ground_truth_coords'][:]
                self.gt_q_all   = f['ground_truth_quality'][:]
            self.preload_mode = True
        else:
            print(f"[*] Dataset size ({total_expected_bytes / (1024**3):.2f} GB) "
                  "exceeds safe RAM limits. Streaming from HDF5...")
            self.preload_mode = False
 
    def get_generator(self):
        import h5py
 
        total_samples = self.total_samples
        batch_size    = self.batch_size
 
        def _make_batch(ref, search, gt_coords, gt_q):
            inputs  = {"reference_stack": tf.convert_to_tensor(ref,    dtype=tf.float32),
                       "search_frame":    tf.convert_to_tensor(search, dtype=tf.float32)}
            targets = {"ground_truth_coords": tf.convert_to_tensor(gt_coords, dtype=tf.float32),
                       "ground_truth_quality": tf.convert_to_tensor(gt_q,  dtype=tf.float32)}
            return inputs, targets
 
        if self.preload_mode:
            def generator():
                indices = np.arange(total_samples)
                np.random.RandomState(42).shuffle(indices)
                num_batches = total_samples // batch_size
                for idx in range(num_batches):
                    bi = indices[idx * batch_size: (idx + 1) * batch_size]
                    yield _make_batch(
                        self.ref_all[bi].astype(np.float32),
                        self.search_all[bi].astype(np.float32),
                        self.gt_coords_all[bi].astype(np.float32),
                        self.gt_q_all[bi].astype(np.float32),
                    )
        else:
            def generator():
                indices = np.arange(total_samples)
                np.random.RandomState(42).shuffle(indices)
                num_batches = total_samples // batch_size
                with h5py.File(self.h5_path, 'r') as f:
                    for idx in range(num_batches):
                        bi          = indices[idx * batch_size: (idx + 1) * batch_size]
                        sorted_ord  = np.argsort(bi)
                        sorted_bi   = bi[sorted_ord]
                        rev_ord     = np.argsort(sorted_ord)
                        yield _make_batch(
                            f['reference_stack'][sorted_bi][rev_ord].astype(np.float32),
                            f['search_frame'][sorted_bi][rev_ord].astype(np.float32),
                            f['ground_truth_coords'][sorted_bi][rev_ord].astype(np.float32),
                            f['ground_truth_quality'][sorted_bi][rev_ord].astype(np.float32),
                        )

        return generator()


def load_memory_safe_dataset(h5_path, batch_size=16):
    dataset_obj = MemorySafeDataset(h5_path, batch_size)
    return dataset_obj, dataset_obj.total_samples


# =====================================================================
# QAT config class
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomLayerQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):      return []
    def get_activations_and_quantizers(self, layer):  return []
    def set_quantize_weights(self, layer, qw):        pass
    def set_quantize_activations(self, layer, qa):    pass
    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False
        )]
    def get_config(self): return {}


# =====================================================================
# Loss registry — maps lowercase alias → callable that returns a Loss instance.
# Extend this dict to support additional loss functions.
# =====================================================================

_LOSS_REGISTRY = {
    # Aliases for SoftArgmaxCoordLoss
    "softargmaxcoordloss":       lambda: SoftArgmaxCoordLoss(beta=30.0, huber_delta=1.0, peak_threshold=0.1),
    "soft_argmax_coord_loss":    lambda: SoftArgmaxCoordLoss(beta=30.0, huber_delta=1.0, peak_threshold=0.1),
    "soft_argmax_coord":         lambda: SoftArgmaxCoordLoss(beta=30.0, huber_delta=1.0, peak_threshold=0.1),
    # Standard Keras losses
    "mse":                       lambda: tf.keras.losses.MeanSquaredError(),
    "meansquarederror":          lambda: tf.keras.losses.MeanSquaredError(),
    "mae":                       lambda: tf.keras.losses.MeanAbsoluteError(),
    "meanabsoluteerror":         lambda: tf.keras.losses.MeanAbsoluteError(),
    "bce":                       lambda: tf.keras.losses.BinaryCrossentropy(),
    "binarycrossentropy":        lambda: tf.keras.losses.BinaryCrossentropy(),
    "huber":                     lambda: tf.keras.losses.Huber(delta=1.0),
    "cce":                       lambda: tf.keras.losses.CategoricalCrossentropy(),
    "categoricalcrossentropy":   lambda: tf.keras.losses.CategoricalCrossentropy(),
}


def _resolve_loss(loss_alias: str) -> tf.keras.losses.Loss:
    """Look up a loss by alias (case-insensitive). Raises SystemExit on unknown alias."""
    key = loss_alias.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _LOSS_REGISTRY:
        print(
            f"[ERROR] Unknown loss name '{loss_alias}'.\n"
            f"        Available losses: {sorted(_LOSS_REGISTRY.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    return _LOSS_REGISTRY[key]()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Apply QAT to Tracker Ver 4 Keras model.")
    parser.add_argument("--keras_in",   required=True,
                        help="Path to input pre-trained float32 Keras model (.keras)")
    parser.add_argument("--keras_out",  required=True,
                        help="Path to save QAT-optimized Keras model")
    parser.add_argument("--train_mode", default="teacher-student",
                        choices=["teacher-student", "dataset_labels"],
                        help="QAT training mode: teacher-student (distillation) "
                             "or dataset_labels (direct targets)")
    parser.add_argument("--h5_dataset", required=True,
                        help="Path to compiled dataset.h5 file")
    parser.add_argument("--epochs",     type=int, default=2,
                        help="Number of QAT fine-tuning epochs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for fine-tuning (default: 16)")
    parser.add_argument("--lr",         type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per epoch (default: all)")
    parser.add_argument(
        "--output_ops",
        default=None,
        help=(
            "Comma-separated list of 'output_name:loss_name' pairs specifying "
            "which model outputs to optimize and with which loss function. "
            "Only listed outputs contribute to the loss; variables of other output "
            "branches are frozen. Both the output name and the loss name are "
            "validated at runtime — an unknown value causes an immediate error.\n"
            "If not set, all model outputs are optimized using MSE.\n\n"
            "Format:  output_name:loss_name[,output_name:loss_name,...]\n"
            "Example: predicted_heatmap:SoftArgmaxCoordLoss\n"
            "Example: predicted_heatmap:SoftArgmaxCoordLoss,predicted_quality:bce\n\n"
            "Available loss names (case-insensitive):\n"
            "  SoftArgmaxCoordLoss / soft_argmax_coord_loss / soft_argmax_coord\n"
            "      Differentiable soft-argmax (beta=30) extracts [y,x] coords\n"
            "      from the predicted heatmap; Huber loss (delta=1.0) vs GT peak.\n"
            "      Positive-sample mask applied (target peak threshold=0.1).\n"
            "  mse / MeanSquaredError      — pixel-wise mean squared error\n"
            "  mae / MeanAbsoluteError     — pixel-wise mean absolute error\n"
            "  bce / BinaryCrossentropy    — binary cross-entropy (sigmoid)\n"
            "  huber                       — Huber loss (delta=1.0)\n"
            "  cce / CategoricalCrossentropy\n"
        ),
    )
    parser.add_argument(
        "--student_init_keras",
        default=None,
        help=(
            "Path to a previously saved QAT Keras model (.keras) to resume training from. "
            "When provided, the QAT model is loaded directly (skipping annotate/quantize_apply) "
            "and fine-tuning continues from its weights with the specified --lr. "
            "--keras_in is still required as the float32 teacher for distillation."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Imports and environment checks
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.dirname(script_dir))

    try:
        import tensorflow_model_optimization as tfmot  # noqa: F811 (re-import after path setup)
    except ImportError:
        print("[ERROR] tensorflow_model_optimization is not installed.", file=sys.stderr)
        sys.exit(1)

    try:
        import tracker_model
    except ImportError:
        print("[ERROR] Could not import tracker_model.py.", file=sys.stderr)
        sys.exit(1)

    # Restore original Conv2D / DepthwiseConv2D if tracker_model overrode them
    import tensorflow.keras.layers as keras_layers
    from tensorflow_model_optimization.python.core.keras.compat import keras as tfmot_keras
    if hasattr(tracker_model, '_orig_conv2d'):
        tf.keras.layers.Conv2D = tracker_model._orig_conv2d
        keras_layers.Conv2D    = tracker_model._orig_conv2d
        tfmot_keras.layers.Conv2D = tracker_model._orig_conv2d
    if hasattr(tracker_model, '_orig_dw_conv2d'):
        tf.keras.layers.DepthwiseConv2D = tracker_model._orig_dw_conv2d
        keras_layers.DepthwiseConv2D    = tracker_model._orig_dw_conv2d
        tfmot_keras.layers.DepthwiseConv2D = tracker_model._orig_dw_conv2d

    # ------------------------------------------------------------------
    # 2. Load the float32 teacher model (always required)
    # ------------------------------------------------------------------
    print(f"[*] Loading teacher (float32) model from: {args.keras_in} ...")
    custom_objects = {
        "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
        "DepthToSpace":               tracker_model.DepthToSpace,
        "HeatmapNormalization":       tracker_model.HeatmapNormalization,
        "CustomLayerQuantizeConfig":  CustomLayerQuantizeConfig,
    }

    with tfmot.quantization.keras.quantize_scope(custom_objects):
        teacher_model = tf.keras.models.load_model(
            args.keras_in, compile=False, safe_mode=False,
            custom_objects=custom_objects,
        )

        # 3. Build or load the QAT student model
        if args.student_init_keras:
            # ------------------------------------------------------------------
            # Resume path: build a clean QAT architecture from the teacher
            # (same annotate/quantize_apply flow), then transplant weights from
            # the saved student checkpoint via load_weights(skip_mismatch=True).
            #
            # Why not load_model directly?
            #   A .keras file saved WITH optimizer state embeds extra variables
            #   (e.g. optimizer_step) that cause a variable-count mismatch when
            #   reloading with compile=False.  Building fresh + load_weights
            #   sidesteps this entirely.
            # ------------------------------------------------------------------
            print(f"[*] Resuming from existing QAT student model: {args.student_init_keras} ...")
            if not os.path.exists(args.student_init_keras):
                print(
                    f"[ERROR] --student_init_keras path not found: {args.student_init_keras}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print("[*] Annotating model layers for QAT (fresh architecture for weight transplant)...")

            def annotate_layer(layer):
                if any(k in layer.name for k in ["quality", "predicted_quality"]):
                    return layer
                if layer.__class__.__name__ in ("Conv2D", "DepthwiseConv2D", "Dense"):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                if layer.__class__.__name__ in (
                    "DepthwiseCorrelationFusion", "DepthToSpace", "HeatmapNormalization",
                    "UpSampling2D", "AveragePooling2D", "MaxPooling2D", "Concatenate", "Add",
                ):
                    return tfmot.quantization.keras.quantize_annotate_layer(
                        layer, CustomLayerQuantizeConfig()
                    )
                if isinstance(layer, tf.keras.Model):
                    return tf.keras.models.clone_model(layer, clone_function=annotate_layer)
                return layer

            qat_annotated = tf.keras.models.clone_model(
                teacher_model, clone_function=annotate_layer
            )
            qat_annotated.set_weights(teacher_model.get_weights())
            qat_model = tfmot.quantization.keras.quantize_apply(qat_annotated)

            # Transplant student weights; skip_mismatch=True ignores any
            # optimizer-state variables that were baked into the saved file.
            print(f"[*] Loading student weights (skip_mismatch=True): {args.student_init_keras} ...")
            qat_model.load_weights(args.student_init_keras, skip_mismatch=True)
            print("[*] Student weights loaded successfully (optimizer state skipped).")
        else:
            # ------------------------------------------------------------------
            # Fresh path: annotate and wrap the teacher with QAT from scratch.
            # ------------------------------------------------------------------
            print("[*] Annotating model layers for QAT...")

            def annotate_layer(layer):
                if any(k in layer.name for k in ["quality", "predicted_quality"]):
                    return layer
                if layer.__class__.__name__ in ("Conv2D", "DepthwiseConv2D", "Dense"):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                if layer.__class__.__name__ in (
                    "DepthwiseCorrelationFusion", "DepthToSpace", "HeatmapNormalization",
                    "UpSampling2D", "AveragePooling2D", "MaxPooling2D", "Concatenate", "Add",
                ):
                    return tfmot.quantization.keras.quantize_annotate_layer(
                        layer, CustomLayerQuantizeConfig()
                    )
                if isinstance(layer, tf.keras.Model):
                    return tf.keras.models.clone_model(layer, clone_function=annotate_layer)
                return layer

            qat_annotated = tf.keras.models.clone_model(
                teacher_model, clone_function=annotate_layer
            )
            print("[*] Transferring pre-trained weights to annotated model...")
            qat_annotated.set_weights(teacher_model.get_weights())

            print("[*] Applying quantization wrappers...")
            qat_model = tfmot.quantization.keras.quantize_apply(qat_annotated)

    # ------------------------------------------------------------------
    # 4. Resolve output_ops against the model's actual output names
    # ------------------------------------------------------------------
    # tfmot.quantize_apply() adds a 'quant_' prefix to wrapped output layer
    # names.  Strip it so the canonical names match the original model and
    # the values the user passes via --output_ops.
    model_output_names = [
        n[len("quant_"):] if n.startswith("quant_") else n
        for n in qat_model.output_names
    ]
    print(f"[*] Model output names: {model_output_names}")

    # ------------------------------------------------------------------
    # 5. Parse output_ops and build per-output loss function list
    #
    #    Format: "output_name:loss_name[,output_name:loss_name,...]"
    #    Both output name and loss name are validated; any unknown value
    #    causes sys.exit(1) with a clear message.
    #    If --output_ops is not set, all outputs use MSE.
    # ------------------------------------------------------------------

    # Each element is (output_name, loss_instance) for active outputs.
    active_output_pairs: list  # list of (str, tf.keras.losses.Loss)

    if args.output_ops:
        entries = [e.strip() for e in args.output_ops.split(",") if e.strip()]
        active_output_pairs = []
        for entry in entries:
            if ":" not in entry:
                print(
                    f"[ERROR] Missing loss specification for '{entry}'.\n"
                    f"        Use format 'output_name:loss_name'\n"
                    f"        Example: predicted_heatmap:SoftArgmaxCoordLoss",
                    file=sys.stderr,
                )
                sys.exit(1)
            output_name, loss_alias = entry.split(":", 1)
            output_name = output_name.strip()
            loss_alias  = loss_alias.strip()
            if output_name not in model_output_names:
                print(
                    f"[ERROR] Output name '{output_name}' not found in model.\n"
                    f"        Available: {model_output_names}",
                    file=sys.stderr,
                )
                sys.exit(1)
            loss_fn = _resolve_loss(loss_alias)
            active_output_pairs.append((output_name, loss_fn))
    else:
        # Default: optimize all outputs with MSE (explicit, no guessing)
        active_output_pairs = [
            (name, _resolve_loss("mse")) for name in model_output_names
        ]

    active_output_names   = [name for name, _ in active_output_pairs]
    active_output_indices = [model_output_names.index(n) for n in active_output_names]
    inactive_output_names = [n for n in model_output_names if n not in active_output_names]

    # Build loss fn list aligned with model_output_names by index
    # (inactive positions hold a placeholder — never called during training)
    output_loss_fns = [None] * len(model_output_names)
    for name, loss_fn in active_output_pairs:
        output_loss_fns[model_output_names.index(name)] = loss_fn

    print(f"[*] Active output ops:")
    for name, loss_fn in active_output_pairs:
        print(f"      '{name}' → {loss_fn.name}")
    if inactive_output_names:
        print(f"[*] Frozen output ops: {inactive_output_names} (excluded from gradients)")

    # ------------------------------------------------------------------
    # 6. Variable exclusion: fragments derived from inactive output names
    # ------------------------------------------------------------------
    # Variables whose name contains any inactive output name fragment are
    # excluded from gradient updates (the branch is effectively frozen).
    _exclude_var_keys = list(inactive_output_names)

    # ------------------------------------------------------------------
    # 7. Optimizer
    # ------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, jit_compile=False)

    # ------------------------------------------------------------------
    # 8. Training step functions
    #    The for-loops below iterate over Python lists (active_output_indices,
    #    output_loss_fns) — they are unrolled at @tf.function trace time and
    #    produce no Python-level overhead during training.
    # ------------------------------------------------------------------

    @tf.function
    def train_step_distill(ref, search):
        # Teacher inference (frozen float32)
        t_outputs = teacher_model([ref, search], training=False)

        with tf.GradientTape() as tape:
            s_outputs = qat_model([ref, search], training=True)

            total_loss  = tf.constant(0.0, dtype=tf.float32)
            per_out_loss = [tf.constant(0.0, dtype=tf.float32)] * len(model_output_names)

            for idx in active_output_indices:
                t_out    = tf.stop_gradient(t_outputs[idx])
                s_out    = s_outputs[idx]
                out_loss = output_loss_fns[idx](t_out, s_out)
                per_out_loss[idx] = out_loss
                total_loss = total_loss + out_loss

        trainable_vars = [
            v for v in qat_model.trainable_variables
            if not any(k in v.name for k in _exclude_var_keys)
        ]
        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        # Return total + per-output losses (first two for display compatibility)
        loss_0 = per_out_loss[0] if len(per_out_loss) > 0 else tf.constant(0.0)
        loss_1 = per_out_loss[1] if len(per_out_loss) > 1 else tf.constant(0.0)
        return total_loss, loss_0, loss_1

    @tf.function
    def train_step_gt(ref, search, gt_targets):
        """gt_targets: list of tensors aligned with model_output_names."""
        with tf.GradientTape() as tape:
            s_outputs = qat_model([ref, search], training=True)

            total_loss   = tf.constant(0.0, dtype=tf.float32)
            per_out_loss = [tf.constant(0.0, dtype=tf.float32)] * len(model_output_names)

            for idx in active_output_indices:
                out_loss = output_loss_fns[idx](gt_targets[idx], s_outputs[idx])
                per_out_loss[idx] = out_loss
                total_loss = total_loss + out_loss

        trainable_vars = [
            v for v in qat_model.trainable_variables
            if not any(k in v.name for k in _exclude_var_keys)
        ]
        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        loss_0 = per_out_loss[0] if len(per_out_loss) > 0 else tf.constant(0.0)
        loss_1 = per_out_loss[1] if len(per_out_loss) > 1 else tf.constant(0.0)
        return total_loss, loss_0, loss_1

    # ------------------------------------------------------------------
    # 9. Load dataset
    # ------------------------------------------------------------------
    print(f"[*] Loading calibration dataset from: {args.h5_dataset} ...")
    dataset_loader, num_samples = load_memory_safe_dataset(args.h5_dataset, args.batch_size)

    # ------------------------------------------------------------------
    # 10. Fine-tuning loop
    # ------------------------------------------------------------------
    print(f"[*] Starting QAT fine-tuning (Mode: {args.train_mode}, Epochs: {args.epochs})...")

    # Dataset key order aligned with model_output_names for dataset_labels mode
    _ds_keys = {
        "ground_truth_heatmap": None,
        "ground_truth_quality": None,
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss_accum = 0.0
        steps = 0

        epoch_dataset    = dataset_loader.get_generator()
        steps_per_epoch  = num_samples // args.batch_size
        if args.max_samples is not None:
            steps_per_epoch = max(1, min(
                steps_per_epoch,
                (args.max_samples + args.batch_size - 1) // args.batch_size,
            ))

        pbar = tqdm.tqdm(epoch_dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            if steps >= steps_per_epoch:
                break

            inputs, targets = batch
            ref    = inputs["reference_stack"]
            search = inputs["search_frame"]

            if args.train_mode == "teacher-student":
                loss_val, l_0, l_1 = train_step_distill(ref, search)
            else:
                # Build gt_targets list aligned with model_output_names index order.
                # Mapping: ground_truth_heatmap → predicted_heatmap (index 0),
                #          ground_truth_quality  → predicted_quality  (index 1).
                # This mapping uses the dataset keys and is the only place that
                # knows about dataset structure — isolated here intentionally.
                _gt_by_name = {
                    "predicted_heatmap": targets["ground_truth_heatmap"],
                    "predicted_quality": targets["ground_truth_quality"],
                }
                gt_targets = [_gt_by_name.get(n, tf.zeros([1])) for n in model_output_names]
                loss_val, l_0, l_1 = train_step_gt(ref, search, gt_targets)

            loss_val_np = float(loss_val)
            total_loss_accum += loss_val_np
            steps += 1

            postfix = {"loss": f"{loss_val_np:.6f}"}
            if len(model_output_names) > 0:
                postfix[f"{active_output_names[0][:8]}_loss"] = f"{float(l_0):.6f}"
            if len(active_output_names) > 1:
                postfix[f"{active_output_names[1][:8]}_loss"] = f"{float(l_1):.6f}"
            pbar.set_postfix(postfix)

            del inputs, targets, batch, ref, search
            del loss_val, l_0, l_1
            if steps % 100 == 0:
                import gc
                gc.collect()

        print(f"--> Epoch {epoch+1} Average Loss: {total_loss_accum / steps:.6f}")

    # ------------------------------------------------------------------
    # 11. Save
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(args.keras_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    qat_model.optimizer = None
    qat_model.save(args.keras_out)
    print(f"[SUCCESS] QAT-optimized model saved to: {args.keras_out}")


if __name__ == "__main__":
    main()
