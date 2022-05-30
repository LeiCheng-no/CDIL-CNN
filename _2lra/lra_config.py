config = {
    "image": {
        "training": {
            "batch_size": 32,
            "epoch": 100
        },
        "models": {
                "n_class": 10,
                "n_length": 1024,
                "fix_length": True,
                "use_embedding": True,
                "vocab_size": 256,
                "dim": 64,
                "cnn_layer": 9,
                "cnn_hidden": 64,
                "cnn_ks": 3,
                "rnn_layer": 1,
                "rnn_hidden": 128
        }
    },
    "pathfinder32": {
        "training": {
            "batch_size": 256,
            "epoch": 100
        },
        "models": {
                "n_class": 2,
                "n_length": 1024,
                "fix_length": True,
                "use_embedding": True,
                "vocab_size": 256,
                "dim": 64,
                "cnn_layer": 9,
                "cnn_hidden": 64,
                "cnn_ks": 3,
                "rnn_layer": 1,
                "rnn_hidden": 128
        }
    },
    "text_4000": {
        "training": {
            "batch_size": 32,
            "epoch": 100
        },
        "models": {
                "n_class": 2,
                "n_length": 4000,
                "fix_length": False,
                "use_embedding": True,
                "vocab_size": 256,
                "dim": 64,
                "cnn_layer": 11,
                "cnn_hidden": 64,
                "cnn_ks": 3,
                "rnn_layer": 1,
                "rnn_hidden": 128
        }
    },
    "retrieval_4000": {
        "training": {
            "batch_size": 256,
            "epoch": 100
        },
        "models": {
            "n_class": 2,
            "n_length": 4000,
            "fix_length": False,
            "use_embedding": True,
            "vocab_size": 256,
            "dim": 64,
            "cnn_layer": 11,
            "cnn_hidden": 64,
            "cnn_ks": 3,
            "rnn_layer": 1,
            "rnn_hidden": 128
        }
    }
}
