# Refer to https://github.com/mlpen/Nystromformer/blob/main/LRA/datasets/text.py

import input_pipeline_text
import pickle
import os

os.makedirs('../lra_datasets/', exist_ok=True)

train_ds, eval_ds, test_ds, encoder = input_pipeline_text.get_tc_datasets(
    n_devices=1, task_name="imdb_reviews", data_dir=None,
    batch_size=1, fixed_vocab=None, max_length=4000)

mapping = {"train": train_ds, "dev": eval_ds, "test": test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0": inst["inputs"].numpy()[0],
            "label": inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end="\r")
    with open(f"../lra_datasets/text_4000.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
