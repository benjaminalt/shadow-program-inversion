"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import json
import os
from argparse import ArgumentParser

from shadow_program_inversion.priors.neural_prior import NeuralPrior
import shadow_program_inversion.utils.config as cfg


def main(args):
    model_config_path = os.path.join(cfg.REPO_DIR, "shadow_program_inversion", "model", "config",
                                     "autoregressive_small.json")
    data_dir = os.path.join(cfg.DATA_DIR, "dmp", args.material, f"sim_train")
    output_dir = os.path.join(cfg.TRAINED_MODELS_DIR, "dmp", args.material)
    with open(model_config_path) as model_config_file:
        model_config = json.load(model_config_file)
    sim = NeuralPrior("Move Linear", 7 + 1, model_config)
    sim.train(data_dir, output_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    main(parser.parse_args())
