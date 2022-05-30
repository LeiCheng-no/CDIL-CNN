# Refer to https://github.com/mlpen/Nystromformer/blob/main/LRA/datasets/retrieval.py

import input_pipeline_retrieval
import pickle
import os

os.makedirs('../lra_datasets/', exist_ok=True)

train_ds, eval_ds, test_ds, encoder = input_pipeline_retrieval.get_matching_datasets(
    n_devices=1, task_name=None, data_dir="./lra_release/tsv_data/",
    batch_size=1, fixed_vocab=None, max_length=4000, tokenizer="char",
    vocab_file_path=None)

mapping = {"train": train_ds, "dev": eval_ds, "test": test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0": inst["inputs1"].numpy()[0],
            "input_ids_1": inst["inputs2"].numpy()[0],
            "label": inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end="\r")
    with open(f"../lra_datasets/retrieval_4000.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
