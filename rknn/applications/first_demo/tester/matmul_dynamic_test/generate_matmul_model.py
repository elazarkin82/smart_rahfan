import os
import tensorflow as tf
from tensorflow.keras import layers, models
import tf2onnx

def main():
    print("Generating MatMul model with dynamic inputs...")
    # Define inputs with shape (128, 64) and (64, 128) - batch size 1
    input_A = layers.Input(shape=(128, 64), batch_size=1, name="input_A")
    input_B = layers.Input(shape=(64, 128), batch_size=1, name="input_B")
    
    # Perform matrix multiplication
    output = layers.Lambda(lambda x: tf.matmul(x[0], x[1]), name="matmul_output")([input_A, input_B])
    
    model = models.Model(inputs=[input_A, input_B], outputs=output)
    model.summary()
    
    # Save the model
    keras_path = "matmul.keras"
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")
    
    # Convert to ONNX directly
    spec_A = tf.TensorSpec((1, 128, 64), tf.float32, name="input_A")
    spec_B = tf.TensorSpec((1, 64, 128), tf.float32, name="input_B")
    
    onnx_path = "matmul.onnx"
    print(f"Converting Keras model to ONNX: {onnx_path}...")
    tf2onnx.convert.from_keras(
        model,
        input_signature=(spec_A, spec_B),
        opset=13,
        output_path=onnx_path
    )
    print(f"[SUCCESS] Saved ONNX model to {onnx_path}")

if __name__ == "__main__":
    main()
