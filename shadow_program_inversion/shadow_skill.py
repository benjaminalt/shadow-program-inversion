import datetime
import json
import os
import time
from typing import Tuple

import torch
import numpy as np
from pytorch_utils import transformations
from pytorch_utils.utils import differentiable_len

from torch.utils.data import DataLoader
from tqdm import tqdm

from shadow_program_inversion.data.dataset import DirectoryDataset
from shadow_program_inversion.model.residual_gru import ResidualGRU
from shadow_program_inversion.priors.differentiable_prior import DifferentiablePrior
from shadow_program_inversion.priors.static_prior import StaticPrior
from shadow_program_inversion.utils.io import load_data_file

torch.set_printoptions(precision=5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ShadowSkill(object):
    def __init__(self, template_type: str, static_input_size: int, model_config: dict,
                 simulator: DifferentiablePrior):
        self.template_type = template_type
        self.static_input_size = static_input_size
        self.model_config = model_config
        self.simulator = simulator
        self.model = None
        self.input_limits = None
        self.output_limits = None
        self.learnable_parameter_gradient_mask = torch.ones(static_input_size, dtype=torch.float32, device=device)

    @staticmethod
    def load(input_dir: str, simulator: DifferentiablePrior):
        print("Loading Neural Template from {}".format(input_dir))
        with open(os.path.join(input_dir, "model_config.json")) as config_file:
            model_config = json.load(config_file)
        with open(os.path.join(input_dir, "misc.json")) as config_file:
            misc = json.load(config_file)
            static_input_size = misc["static_input_size"]
            template_type = misc["template_type"]

        template = ShadowSkill(template_type, static_input_size, model_config, simulator)
        template.model = ResidualGRU.load(os.path.join(input_dir, "model.pt"), device)
        with open(os.path.join(input_dir, "limits.json")) as config_file:
            limits = json.load(config_file)
            template.input_limits = torch.tensor(limits["inputs"]).to(device)
            template.output_limits = torch.tensor(limits["outputs"]).to(device)
        return template

    def _save(self, output_dir, training_history=None):
        out_dir = os.path.join(output_dir, "{}_({})".format(type(self).__name__, time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(out_dir)
        # Model
        self.model.save(os.path.join(out_dir, "model.pt"))
        # Model config
        with open(os.path.join(out_dir, "model_config.json"), "w") as json_file:
            json.dump(self.model_config, json_file)
        # Input and output limits
        with open(os.path.join(out_dir, "limits.json"), "w") as json_file:
            json.dump({
                "inputs": self.input_limits.cpu().tolist(),
                "outputs": self.output_limits.cpu().tolist()}, json_file)
        # Misc settings
        with open(os.path.join(out_dir, "misc.json"), "w") as json_file:
            json.dump({"static_input_size": self.static_input_size, "template_type": self.template_type}, json_file)
        # Training history
        if training_history is not None:
            with open(os.path.join(out_dir, "training_history.json"), "w") as training_history_file:
                json.dump(training_history, training_history_file)
        print("Results saved under {}".format(out_dir))

    @staticmethod
    def load_data(data_dir, num_data: int = None):
        all_inputs = []
        all_points_from = []
        all_sim = []
        all_real = []
        total_data = 0
        for data_filename in os.listdir(data_dir):
            inputs, points_from, sim, real = load_data_file(os.path.join(data_dir, data_filename))
            total_data += len(inputs)
            all_inputs.append(inputs)
            all_points_from.append(points_from)
            if sim is not None:
                all_sim.append(sim)
            if real is not None:
                all_real.append(real)
        num_data = num_data if num_data is not None else total_data
        all_inputs = torch.cat(all_inputs, dim=0)[:num_data]
        all_points_from = torch.cat(all_points_from, dim=0)[:num_data]
        all_sim = torch.cat(all_sim, dim=0)[:num_data] if len(all_sim) > 0 else None
        all_real = torch.cat(all_real, dim=0)[:num_data] if len(all_real) > 0 else None
        return all_inputs, all_points_from, all_sim, all_real

    def train(self, data_dir: str, output_dir: str, use_simulator=False):
        all_data = DirectoryDataset(data_dir)

        self._set_input_output_limits(all_data)
        self.input_limits = self.input_limits.to(device)
        self.output_limits = self.output_limits.to(device)
        output_size = self.output_limits.size(-1)

        print("Training neural template of type {} on device '{}'".format(self.template_type, device))

        split_idx = int(0.9 * len(all_data))
        training_data = DirectoryDataset(data_dir, end=split_idx)
        validation_data = DirectoryDataset(data_dir, start=split_idx)
        print("Training/validation split: {}/{}".format(len(training_data), len(validation_data)))
        train_loader = DataLoader(training_data, batch_size=self.model_config["batch_size"], drop_last=True,
                                  pin_memory=True, num_workers=4)
        validate_loader = DataLoader(validation_data, batch_size=self.model_config["batch_size"],
                                     drop_last=False, pin_memory=True, num_workers=1)

        if self.model is not None:
            print("Model is already trained. Fine-tuning...")
            self.model = self.model.to(device)
        else:
            self.model = ResidualGRU(input_size=self.static_input_size + output_size,
                                     output_size=output_size,
                                     hidden_size=self.model_config["hidden_size"],
                                     num_layers=self.model_config["num_layers"],
                                     dropout_p=self.model_config["dropout_p"]).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config["learning_rate"], weight_decay=1e-5)
        loss_functions = {
            "eos_probability": [torch.nn.BCELoss(), 1],
            "success_probability": [torch.nn.BCELoss(), 1],
            "trajectory": [torch.nn.MSELoss(), 10]
        }
        training_history = []
        best_weights = None
        best_validation_loss = np.inf

        def unpack_data(data: Tuple[torch.Tensor]):
            inp, start_states, sim, real = data
            inp = inp.to(device, non_blocking=True)
            start_states = start_states.to(device, non_blocking=True)
            sim = sim.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)
            if use_simulator:
                sim = self.simulator.simulate(inp, start_states, max_trajectory_len=real.size(1))
            return inp, start_states, sim, real

        print("Training for {} epochs with batch size {}".format(self.model_config["epochs"],
                                                                 self.model_config["batch_size"]))
        start = time.time()
        for epoch in range(self.model_config["epochs"]):
            print('{} Epoch {}/{}'.format("#" * 10, epoch + 1, self.model_config["epochs"]))
            epoch_start_time = time.time()
            total_train_losses = [0] * 4
            total_validation_losses = [0] * 4

            self.model.train()
            for data in tqdm(train_loader):
                static_inputs, start_states, sim, real = unpack_data(data)
                losses = self._training_step(static_inputs, sim, real, optimizer, loss_functions, evaluate=False)
                for i in range(len(losses)):
                    total_train_losses[i] += losses[i]
            avg_train_losses = list(map(lambda x: x / len(train_loader), total_train_losses))

            self.model.eval()
            with torch.no_grad():
                for data in tqdm(validate_loader):
                    static_inputs, start_states, sim, real = unpack_data(data)
                    losses = self._training_step(static_inputs, sim, real, optimizer, loss_functions, evaluate=True)
                    for i in range(len(losses)):
                        total_validation_losses[i] += losses[i]
                avg_validation_losses = list(map(lambda x: x / len(validate_loader), total_validation_losses))
                if avg_validation_losses[0] < best_validation_loss:
                    best_weights = self.model.state_dict()

            training_history.append([avg_train_losses, avg_validation_losses])
            print("Avg loss: Train: {:.6f} ({:.6f} {:.6f} {:.6f}), validation: {:.6f} ({:.6f} {:.6f} {:.6f})".format(
                *avg_train_losses, *avg_validation_losses))

            print("Epoch {} took {:.2f}s".format(epoch + 1, time.time() - epoch_start_time))

        total_training_time = time.time() - start
        print("Training took {}".format(str(datetime.timedelta(seconds=total_training_time)).split(".")[0]))
        print("Setting model weights to minimize validation loss")
        self.model.load_state_dict(best_weights)
        self._save(output_dir, training_history)

    def _training_step(self, static_inputs_world: torch.Tensor, sim_world: torch.Tensor, real_world: torch.Tensor,
                       optimizer, loss_functions, evaluate=False):
        optimizer.zero_grad()
        point_start_world = real_world[:, 0, 2:9]
        normalized_real = normalize_labels(real_world, self.output_limits)
        normalized_sim = normalize_labels(sim_world, self.output_limits)
        residual_label = compute_residual(normalized_real, normalized_sim)

        _, _, normalized_residual = self.atomic_forward(static_inputs_world, point_start_world, sim_world,
                                                        denormalize_out=False)

        eos_loss = loss_functions["eos_probability"][0](normalized_residual[:, :, 0], residual_label[:, :, 0]) * \
                   loss_functions["eos_probability"][1]
        success_loss = loss_functions["success_probability"][0](normalized_residual[:, :, 1], residual_label[:, :, 1]) * \
                       loss_functions["success_probability"][1]
        trajectory_loss = loss_functions["trajectory"][0](normalized_residual[:, :, 2:], residual_label[:, :, 2:]) * \
                          loss_functions["trajectory"][1]
        loss = trajectory_loss + eos_loss + success_loss

        if not evaluate:
            loss.backward()
            optimizer.step()

        return loss.item(), trajectory_loss.item(), eos_loss.item(), success_loss.item()

    def atomic_forward(self, static_inputs_world: torch.Tensor, point_start_world: torch.Tensor,
                       simulation_world: torch.Tensor = None, denormalize_out: bool = False,
                       cache_sim: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(static_inputs_world.size()) == len(point_start_world.size()) == 2, "Batches required"
        batch_size = static_inputs_world.size(0)

        if simulation_world is None:
            if type(self.simulator) == StaticPrior:
                simulation_world = self.simulator.simulate(static_inputs_world, point_start_world, cache=cache_sim).to(
                    device)
            else:  # Neural simulator
                simulation_world = self.simulator.simulate(static_inputs_world, point_start_world).to(device)

        self.model = self.model.to(device)
        static_inputs_world = static_inputs_world.to(device)

        normalized_sim = normalize_labels(simulation_world, self.output_limits)
        normalized_inputs = normalize_inputs(static_inputs_world, self.input_limits)
        input_seq = normalized_inputs.unsqueeze(1).repeat(1, normalized_sim.size(1), 1)
        combined_input_seq = torch.cat((input_seq, normalized_sim), dim=-1)

        hidden = self.model.init_hidden(batch_size, device)

        normalized_residual = self.model(combined_input_seq, hidden)
        trajectory = apply_residual(normalized_residual, normalized_sim)

        if denormalize_out:
            trajectory = denormalize_outputs(trajectory, self.output_limits)
        return trajectory, simulation_world, normalized_residual

    def predict(self, inputs_world: torch.Tensor, point_start_world: torch.Tensor, max_trajectory_len: int = 500,
                with_grad: bool = False) -> tuple:
        if self.model is None or self.input_limits is None or self.output_limits is None:
            raise RuntimeError("Cannot predict: Model has not been trained")
        self.model.eval()

        has_batch_dim = len(inputs_world.size()) > 1
        if not has_batch_dim:
            inputs_world = inputs_world.unsqueeze(0)
            point_start_world = point_start_world.unsqueeze(0)

        if not with_grad:
            with torch.no_grad():
                trajectory_world, simulation, normalized_residual = self.atomic_forward(inputs_world, point_start_world,
                                                                                        simulation_world=None,
                                                                                        denormalize_out=True)
        else:
            trajectory_world, simulation, normalized_residual = self.atomic_forward(inputs_world, point_start_world,
                                                                                    simulation_world=None,
                                                                                    denormalize_out=True)

        success_probability = self.success_probability(trajectory_world)

        if not has_batch_dim:  # Remove additional batch dim of size 1
            trajectory_world = trajectory_world.reshape(trajectory_world.size(1), trajectory_world.size(2))
            # success_probability already has the right shape
            simulation = simulation.reshape(simulation.size(1), simulation.size(2))

        return trajectory_world, simulation, success_probability

    @staticmethod
    def success_probability(trajectory: torch.Tensor) -> torch.Tensor:
        has_batch_dim = len(trajectory.size()) == 3
        if not has_batch_dim:
            trajectory = trajectory.unsqueeze(0)
        return torch.mean(trajectory[:, :, 1], dim=1)

    @staticmethod
    def trajectory_length(trajectory: torch.Tensor) -> torch.Tensor:
        if type(trajectory) == np.ndarray:
            if len(trajectory.shape) > 2:
                raise NotImplementedError("trajectory_lengths expects non-batched numpy arrays")
            eos_index = np.nonzero(trajectory[:, 0])[0]
            if len(eos_index) > 0:
                return eos_index[0]
            return len(trajectory)
        elif type(trajectory) == torch.Tensor:
            has_batch_dim = len(trajectory.size()) == 3
            if not has_batch_dim:
                trajectory = trajectory.unsqueeze(0)
            lengths = differentiable_len(trajectory)
            return lengths if has_batch_dim else lengths.squeeze()
        else:
            raise NotImplementedError(
                "trajectory_length expects inputs of type ndarray or tensor, not {}".format(type(trajectory)))

    def _set_input_output_limits(self, data: DirectoryDataset):
        if self.input_limits is None and self.output_limits is None:
            print("Setting input and output limits for this NeuralTemplate")

            loader = torch.utils.data.DataLoader(data, batch_size=256, num_workers=4)
            computed_output_limits = None
            computed_input_limits = None

            for inputs, start_state, sim, real in tqdm(loader):
                outputs = torch.cat((sim, real), dim=0)
                batch_output_limits = torch.stack((torch.min(outputs.view(-1, outputs.size(-1)), dim=0)[0],
                                                   torch.max(outputs.view(-1, outputs.size(-1)), dim=0)[0]))
                batch_input_limits = torch.stack((torch.min(inputs, dim=0)[0], torch.max(inputs, dim=0)[0]))
                if computed_input_limits is None:
                    computed_output_limits = batch_output_limits
                    computed_input_limits = batch_input_limits
                else:
                    computed_output_limits = torch.stack((torch.min(batch_output_limits[0], computed_output_limits[0]),
                                                          torch.max(batch_output_limits[1], computed_output_limits[1])))
                    computed_input_limits = torch.stack((torch.min(batch_input_limits[0], computed_input_limits[0]),
                                                         torch.max(batch_input_limits[1], computed_input_limits[1])))

            # If min and max are identical, this causes trouble when scaling --> division by zero produces NaN
            # Set min to val - 1 and max to val + 1
            for dim_idx in range(computed_input_limits.size(1)):
                if computed_input_limits[0, dim_idx] == computed_input_limits[1, dim_idx]:
                    orig_value = computed_input_limits[0, dim_idx].clone()
                    computed_input_limits[0, dim_idx] = orig_value - 1  # new min
                    computed_input_limits[1, dim_idx] = orig_value + 1  # new max
            for dim_idx in range(computed_output_limits.size(1)):
                if computed_output_limits[0, dim_idx] == computed_output_limits[1, dim_idx]:
                    orig_value = computed_output_limits[0, dim_idx].clone()
                    computed_output_limits[0, dim_idx] = orig_value - 1  # new min
                    computed_output_limits[1, dim_idx] = orig_value + 1  # new max

            self.input_limits = computed_input_limits
            self.output_limits = computed_output_limits
        else:
            print("Input and output limits already set")


def compute_residual(real: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    meta_inf = real[:, :, :2]
    residual_trajectory = real[:, :, 2:] - sim[:, :, 2:]
    return torch.cat((meta_inf, residual_trajectory), dim=-1)


def apply_residual(residual: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    meta_inf = residual[:, :, :2]
    complete_trajectory = sim[:, :, 2:] + residual[:, :, 2:]
    return torch.cat((meta_inf, complete_trajectory), dim=-1)


def normalize_inputs(inputs_world: torch.Tensor, input_limits: torch.Tensor) -> torch.Tensor:
    return transformations.scale(inputs_world, input_limits[0], input_limits[1], -1, 1)


def denormalize_inputs(inputs_normalized: torch.Tensor, input_limits: torch.Tensor) -> torch.Tensor:
    return transformations.scale(inputs_normalized, -1, 1, input_limits[0], input_limits[1])


def normalize_labels(labels_world: torch.Tensor, label_limits: torch.Tensor) -> torch.Tensor:
    output_batch = []
    for i, trajectory in enumerate(labels_world):
        unscaled_trajectory = trajectory[:, 2:]
        meta_inf = trajectory[:, :2]
        scaled_trajectory = transformations.scale(unscaled_trajectory, label_limits[0, 2:], label_limits[1, 2:], -1, 1)
        if torch.isnan(scaled_trajectory).any() or torch.isinf(scaled_trajectory).any():
            raise RuntimeError("normalize_labels: Got NaN or Inf after scaling")
        scaled_total = torch.cat((meta_inf, scaled_trajectory), dim=-1)
        output_batch.append(scaled_total)
    return torch.stack(output_batch, dim=0)


def denormalize_outputs(outputs: torch.Tensor, output_limits: torch.Tensor) -> torch.Tensor:
    output_batch = []
    for i, trajectory in enumerate(outputs):
        unscaled_trajectory = trajectory[:, 2:]
        meta_inf = trajectory[:, :2]
        scaled_trajectory = transformations.scale(unscaled_trajectory, -1, 1, output_limits[0, 2:],
                                                  output_limits[1, 2:])
        if torch.isnan(unscaled_trajectory).any() or torch.isinf(unscaled_trajectory).any():
            raise RuntimeError("denormalize_outputs: Got NaN or Inf after scaling")
        scaled_total = torch.cat((meta_inf, scaled_trajectory), dim=-1)
        output_batch.append(scaled_total)
    return torch.stack(output_batch, dim=0)
