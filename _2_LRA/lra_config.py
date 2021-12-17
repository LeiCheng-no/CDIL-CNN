config = {
    "image": {
        "models": {
                "n_class": 10,
                "n_length": 1024,
                'layer_cnn': 9,
                "fix_length": True,
                "use_cuda": True,
                "use_embedding": True,
                "vocab_size": 256,
                "dim": 64,
        },
        "training": {
            "device_id": 0,
            "batch_size": 32,
            "learning_rate": 0.001,
            "eval_frequency": 1,
            "num_train_steps": 50
        }
    },
    "pathfinder32": {
        "models": {
                "n_class": 2,
                "n_length": 1024,
                'layer_cnn': 9,
                "fix_length": True,
                "use_cuda": True,
                "use_embedding": True,
                "vocab_size": 256,
                "dim": 64,
        },
        "training": {
            "device_id": 0,
            "batch_size": 256,
            "learning_rate": 0.001,
            "eval_frequency": 1,
            "num_train_steps": 50
        }
    },
    "listops_2000": {
        "models": {
                "n_class": 10,
                "n_length": 2000,
                'layer_cnn': 10,
                "fix_length": False,
                "use_cuda": True,
                "use_embedding": True,
                "vocab_size": 16,
                "dim": 64,
        },
        "training": {
            "device_id": 0,
            "batch_size": 32,
            "learning_rate": 0.001,
            "eval_frequency": 1,
            "num_train_steps": 50
        }
    },
    "text_4000": {
        "models": {
                "n_class": 2,
                "n_length": 4000,
                'layer_cnn': 11,
                "fix_length": False,
                "use_cuda": True,
                "use_embedding": True,
                "vocab_size": 256,
                "dim": 64,
        },
        "training": {
            "device_id": 0,
            "batch_size": 32,
            "learning_rate": 0.001,
            "eval_frequency": 1,
            "num_train_steps": 50
        }
    },
    "retrieval_4000": {
        "models": {
            "n_class": 2,
            "n_length": 4000,
            'layer_cnn': 11,
            "fix_length": False,
            "use_cuda": True,
            "use_embedding": True,
            "vocab_size": 256,
            "dim": 64,
        },
        "training": {
            "device_id": 0,
            "batch_size": 256,
            "learning_rate": 0.001,
            "eval_frequency": 1,
            "num_train_steps": 50
        }
    }
}
