config = {
        "N_SEEDS": 1,
        "ENV_NAME": "FourRooms-misc",
        "FAIL_PROB": 1.0 / 5.0, 
        "GAMMA": 0.99,
        "k": 32,
        "MAX_STEPS_IN_EPISODE": 1e6,
        "LAYER_NORM": False,

        "LR": 5e-4,
        "LR_END": 5e-4,
        
        "NUM_ENVS": 32,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 5e5,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": 128,
        "GAMMA": 0.99, # similar to that used by RND
        "GAE_LAMBDA": 0.0,
        "CLIP_EPS": 0.2,
        "VF_CLIP": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "NETWORK_TYPE": 'cnn',
        "NORMALIZE_OBS": False,
        "CALC_TRUE_VALUES": True,
        # "OPTIMIZER": "AdamW", # or AdamW
        # "ADAM_EPS": 1e-5,
        # "WEIGHT_DECAY": 0.0
        }