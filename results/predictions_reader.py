import pickle
import numpy as np

for name in ["lidar_mini_predictions.pkl", "radar_mini_predictions.pkl"]:
    with open(f"object_detection/omnihd/{name}", "rb") as f:
        preds = pickle.load(f)

    print(f"=== {name} ===")
    other_keys = [k for k in preds if k != "metadata"]
    print("other top-level keys:", other_keys)

    for k in other_keys:
        v = preds[k]
        print(f"--- key: {k!r} ---")
        print("type:", type(v))
        if isinstance(v, dict):
            print("num entries:", len(v))
            sample_key = next(iter(v))
            print("sample entry key:", sample_key, "(type:", type(sample_key), ")")
            sample_val = v[sample_key]
            print("sample entry value type:", type(sample_val))
            print("sample entry value:", sample_val)
        elif isinstance(v, list):
            print("length:", len(v))
            print("first item type:", type(v[0]))
            print("first item:", v[0])
    print()
