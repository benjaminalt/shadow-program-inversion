import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, os.pardir, os.pardir)

DATA_DIR = os.path.join(REPO_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

OPTIMIZED_PARAM_DIR = os.path.join(REPO_DIR, "optimized_parameters")
if not os.path.exists(OPTIMIZED_PARAM_DIR):
    os.makedirs(OPTIMIZED_PARAM_DIR)

RESULTS_DIR = os.path.join(REPO_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TRAINED_MODELS_DIR = os.path.join(REPO_DIR, "trained_models")
if not os.path.exists(TRAINED_MODELS_DIR):
    os.makedirs(TRAINED_MODELS_DIR)

ROBOT_IP = "172.16.53.128"
