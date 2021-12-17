config = {
    "adding": {
        "dataset": {
            "use_cuda": True,
            "in_dim": 2,
            "n_class": 1,
        },
        "training": {
            "device_id": 0,
            "batch_size": 40,
            "learning_rate": 0.001,
            "eval_frequency": 1,
        },
        "models": {
            "CDIL": {
                "hidden": 32,
                "kernel_size": 3
            },
            "TCN": {
                "hidden": 32,
                "kernel_size": 3
            },
            "LSTM": {
                "layer": 1,
                "hidden": 128,
            },
            "GRU": {
                "layer": 1,
                "hidden": 128,
            },
            "Transformer": {
                "dim": 32,
                "depth": 4,
                "heads": 4,
            },
            "Linformer": {
                "dim": 32,
                "depth": 4,
                "heads": 4,
            },
            "Performer": {
                "dim": 32,
                "depth": 4,
                "heads": 4,
            }
        }
    }
}
