"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import json
import os
from argparse import ArgumentParser

from shadow_program_inversion.priors.neural_prior import NeuralPrior
from shadow_program_inversion.shadow_skill import ShadowSkill
import shadow_program_inversion.utils.config as cfg


def main(args):
    model_dir = os.path.join(cfg.TRAINED_MODELS_DIR, "dmp", args.material)
    prior_dir = os.path.join(model_dir, list(filter(lambda dirname: dirname.startswith("NeuralPrior"), os.listdir(model_dir)))[0])
    sim = NeuralPrior.load(prior_dir)
    model_config_path = os.path.join(cfg.REPO_DIR, "shadow_program_inversion", "model", "config",
                                     "residual_gru.json")
    with open(model_config_path) as model_config_file:
        model_config = json.load(model_config_file)
    shadow_skill = ShadowSkill("Move Linear", static_input_size=8, model_config=model_config, simulator=sim)
    data_dir = os.path.join(cfg.DATA_DIR, "dmp", args.material, "train")
    output_dir = os.path.join(cfg.TRAINED_MODELS_DIR, "dmp", args.material)
    shadow_skill.train(data_dir, output_dir, use_simulator=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    main(parser.parse_args())
