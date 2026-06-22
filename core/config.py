config = {
        "N_SEEDS": 1,
        "ENV_NAME": "FourRooms-misc",
        "FAIL_PROB": 0.001, 
        "GAMMA": 0.99,
        "k": 8,
        "MAX_STEPS_IN_EPISODE": 1e6,
        "LAYER_NORM": True,

        "LR": 5e-4,
        "LR_END": 5e-4,
        
        "NUM_ENVS": 32,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 5e5,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": 128,
        "GAMMA": 0.99, 
        "VALUE_LAMBDA": 0.0,
        "POLICY_LAMBDA": 0.6,
        "GAE_LAMBDA": 0.0,
        "CLIP_EPS": 0.5,
        "VF_CLIP": 0.5,
        "ENT_COEF": 0.00,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "NETWORK_TYPE": 'cnn',
        "NORMALIZE_OBS": False,
        "CALC_TRUE_VALUES": True,
        # IV settings:
        "FORWARD_COEFF": 0.0,
        "REWARD_COEFF": 0.0,
        "DONE_COEFF": 0.0,  
        "POLICY_COEFF": 1.0,
        "V_LOSS_SHARED": True, # whether the TD / MC can update the final weights w.

        # For the fixed:
        'MODEL_LOAD_DIR': "cont",
        }