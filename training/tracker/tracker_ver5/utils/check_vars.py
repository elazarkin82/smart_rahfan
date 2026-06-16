import os
import sys
import tensorflow as tf
import tensorflow_model_optimization as tfmot

project_dir = "/home/elazarkin/work/projects/smart_rahfan/training/tracker/tracker_ver4"
sys.path.append(project_dir)
import tracker_model
from utils.quantization_optimization import CustomLayerQuantizeConfig

custom_objects = {
    "DepthwiseCorrelationFusion": tracker_model.DepthwiseCorrelationFusion,
    "DepthToSpace": tracker_model.DepthToSpace,
    "HeatmapNormalization": tracker_model.HeatmapNormalization,
    "CustomLayerQuantizeConfig": CustomLayerQuantizeConfig,
}

teacher_model = tf.keras.models.load_model("outputs/tracker_coords_fbn.keras", compile=False, safe_mode=False, custom_objects=custom_objects)

def annotate_layer(layer):
    if any(k in layer.name for k in ["quality", "predicted_quality"]):
        return layer
    if layer.__class__.__name__ in ("Conv2D", "DepthwiseConv2D", "Dense"):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    elif layer.__class__.__name__ in (
        "DepthwiseCorrelationFusion",
        "DepthToSpace",
        "HeatmapNormalization",
        "UpSampling2D",
        "AveragePooling2D",
        "MaxPooling2D",
        "Concatenate",
        "Add"
    ):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, CustomLayerQuantizeConfig())
    if isinstance(layer, tf.keras.Model):
        return tf.keras.models.clone_model(layer, clone_function=annotate_layer)
    return layer

qat_annotated = tf.keras.models.clone_model(teacher_model, clone_function=annotate_layer)
qat_annotated.set_weights(teacher_model.get_weights())
qat_model = tfmot.quantization.keras.quantize_apply(qat_annotated)

print("Variables in memory:")
for l in qat_model.layers:
    if "decoder_ir1_expand" in l.name:
        print(f"Layer: {l.name} ({l.__class__.__name__})")
        for v in l.variables:
            print(f"  {v.name}: {v.shape}")
