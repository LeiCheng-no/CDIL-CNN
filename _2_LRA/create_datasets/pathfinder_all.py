# Refer to https://github.com/mlpen/Nystromformer/blob/main/LRA/datasets/pathfinder.py

import numpy as np
import os
import pickle
import tensorflow as tf
import random

root_dir = "./lra_release/"
subdir = "pathfinder32"
ds_list = []
for diff_level in ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]:
    data_dir = os.path.join(root_dir, subdir, diff_level)
    metadata_list = [
        os.path.join(data_dir, "metadata", file)
        for file in os.listdir(os.path.join(data_dir, "metadata"))
        if file.endswith(".npy")
    ]
    for idx, metadata_file in enumerate(metadata_list):
        print(idx, len(metadata_list), metadata_file, "\t\t", end="\r")
        for inst_meta in tf.io.read_file(metadata_file).numpy().decode("utf-8").split("\n")[:-1]:
            metadata = inst_meta.split(" ")
            img_path = os.path.join(data_dir, metadata[0], metadata[1])
            img_bin = tf.io.read_file(img_path)
            if len(img_bin.numpy()) == 0:
                print()
                print("detected empty image")
                continue
            img = tf.image.decode_png(img_bin)
            seq = img.numpy().reshape(-1).astype(np.int32)
            label = int(metadata[3])
            ds_list.append({"input_ids_0": seq, "label": label})

random.shuffle(ds_list)

bp80 = int(len(ds_list) * 0.8)
bp90 = int(len(ds_list) * 0.9)
train = ds_list[:bp80]
dev = ds_list[bp80:bp90]
test = ds_list[bp90:]

with open(f"../lra_datasets/{subdir}.train.pickle", "wb") as f:
    pickle.dump(train, f)
with open(f"../lra_datasets/{subdir}.dev.pickle", "wb") as f:
    pickle.dump(dev, f)
with open(f"../lra_datasets/{subdir}.test.pickle", "wb") as f:
    pickle.dump(test, f)
