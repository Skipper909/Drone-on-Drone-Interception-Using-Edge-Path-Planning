import argparse
import pathlib
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True)
    ap.add_argument("--out", default="policy.tflite")
    ap.add_argument("--mode", choices=["fp32", "fp16", "int8"], default="fp32")
    args = ap.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)

    if args.mode in ("fp16", "int8"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.mode == "fp16":
        converter.target_spec.supported_types = [tf.float16]

    if args.mode == "int8":
        # NOTE: You need representative data for proper INT8.
        # If you don’t have it yet, do fp32/fp16 now and add int8 later.
        def rep_data_gen():
            # Replace with real observations from your sim logs.
            # Shape must match your model input, typically [1, obs_dim].
            import numpy as np
            for _ in range(200):
                yield [np.random.uniform(-1, 1, size=(1, 64)).astype("float32")]
        converter.representative_dataset = rep_data_gen
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite = converter.convert()
    pathlib.Path(args.out).write_bytes(tflite)
    print(f"Wrote {args.out} ({len(tflite)} bytes)")

if __name__ == "__main__":
    main()
