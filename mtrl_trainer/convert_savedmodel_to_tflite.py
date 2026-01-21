import argparse
import pathlib
import tensorflow as tf

# Critical: register TFP TypeSpecs used inside the SavedModel
import tensorflow_probability as tfp  # noqa: F401

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)

    # Conservative default optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.fp16:
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    pathlib.Path(args.out).write_bytes(tflite_model)
    print(f"Wrote {args.out} ({len(tflite_model)} bytes)")

if __name__ == "__main__":
    main()