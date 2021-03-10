"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import datetime
import json
import os
import time

import torch
from pytorch_utils import transformations
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from shadow_program_inversion.data.dataset import DirectoryDataset
from shadow_program_inversion.model.autoregressive import AutoregressiveModel
from shadow_program_inversion.priors.differentiable_prior import DifferentiablePrior
from shadow_program_inversion.priors.static_prior import Group, TrajectoryProperties
from shadow_program_inversion.utils.sequence_utils import delta, undelta

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NeuralPrior(DifferentiablePrior):
    def __init__(self, template_type: str, static_input_size: int, model_config: dict,
                 group: Group = Group.MANIPULATOR, trajectory_properties: TrajectoryProperties = None):
        super(NeuralPrior, self).__init__()
        self.model_config = model_config
        self.template_type = template_type
        self.static_input_size = static_input_size
        self.dynamic_input_size = 7
        self.output_size = 2 + 7 if group == Group.MANIPULATOR else 2 + 1
        self.model = None
        self.input_limits = None
        self.state_limits = None
        self.output_limits = None
        self.group = group
        self.trajectory_properties = trajectory_properties if trajectory_properties is not None else TrajectoryProperties()

    @staticmethod
    def load(input_dir: str):
        print("Loading NeuralSimulator from {}".format(input_dir))
        with open(os.path.join(input_dir, "model_config.json")) as config_file:
            model_config = json.load(config_file)
        with open(os.path.join(input_dir, "misc.json")) as config_file:
            misc = json.load(config_file)
            static_input_size = misc["static_input_size"]
            template_type = misc["template_type"]
            group = Group(misc["group"])
            trajectory_properties = TrajectoryProperties.from_dict(misc["trajectory_properties"])

        simulator = NeuralPrior(template_type, static_input_size, model_config, group, trajectory_properties)
        simulator.model = AutoregressiveModel.load(os.path.join(input_dir, "model.pt"), device)
        with open(os.path.join(input_dir, "limits.json")) as config_file:
            limits = json.load(config_file)
            simulator.input_limits = torch.tensor(limits["inputs"]).to(device)
            simulator.state_limits = torch.tensor(limits["states"]).to(device)
            simulator.output_limits = torch.tensor(limits["outputs"]).to(device)
        return simulator

    def _save(self, output_dir, training_history, checkpoint_nr=None):
        out_dir = os.path.join(output_dir, "{}_({})_{}{}".format(type(self).__name__,
                                                          time.strftime("%Y%m%d-%H%M%S"),
                                                          self.template_type.replace(" ", "_"),
                                                          "_" + str(checkpoint_nr) if checkpoint_nr is not None else ""))
        os.makedirs(out_dir)
        # Model
        self.model.save(os.path.join(out_dir, "model.pt"))
        # Model config
        with open(os.path.join(out_dir, "model_config.json"), "w") as json_file:
            json.dump(self.model_config, json_file)
        # Input and output limits
        with open(os.path.join(out_dir, "limits.json"), "w") as json_file:
            json.dump({"inputs": self.input_limits.cpu().tolist(),
                       "states": self.state_limits.cpu().tolist(),
                       "outputs": self.output_limits.cpu().tolist()},
                      json_file)
        # Misc settings
        with open(os.path.join(out_dir, "misc.json"), "w") as json_file:
            json.dump({"static_input_size": self.static_input_size,
                       "template_type": self.template_type,
                       "group": self.group.value,
                       "trajectory_properties": self.trajectory_properties.to_dict()
                       }, json_file)
        # Training history
        with open(os.path.join(out_dir, "training_history.json"), "w") as training_history_file:
            json.dump(training_history, training_history_file)
        print("Results saved under {}".format(out_dir))

    def train(self, data_dir: str, output_dir: str):
        all_data = DirectoryDataset(data_dir)

        self._set_input_output_limits(all_data)     # NB: Output limits are for delta trajectories
        self.input_limits = self.input_limits.to(device)
        self.output_limits = self.output_limits.to(device)
        self.state_limits = self.state_limits.to(device)

        print("Training neural simulator of type {} on device '{}'".format(self.template_type, device))

        split_idx = int(0.9 * len(all_data))
        training_data = DirectoryDataset(data_dir, end=split_idx)
        validation_data = DirectoryDataset(data_dir, start=split_idx)
        print("Training/validation split: {}/{}".format(len(training_data), len(validation_data)))
        train_loader = DataLoader(training_data, batch_size=self.model_config["batch_size"], drop_last=True,
                                  pin_memory=True, num_workers=4)
        validate_loader = DataLoader(validation_data, batch_size=self.model_config["batch_size"], drop_last=True,
                                     pin_memory=True, num_workers=4)

        if self.model is not None:
            print("Model is already trained. Fine-tuning...")
            self.model = self.model.to(device)
        else:
            self.model = AutoregressiveModel(self.static_input_size + self.dynamic_input_size, self.output_size,
                                             self.model_config["lstm_hidden_size"]).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config["learning_rate"])

        loss_functions = {
            "eos_probability": torch.nn.BCELoss(),
            "success_probability": torch.nn.BCELoss(),
            "trajectory": torch.nn.MSELoss()
        }

        training_history = []
        best_weights = None
        best_validation_loss = np.inf

        print("Training for {} epochs with batch size {}".format(self.model_config["epochs"], self.model_config["batch_size"]))
        start = time.time()
        for epoch in range(self.model_config["epochs"]):
            print('{} Epoch {}/{}'.format("#" * 10, epoch + 1, self.model_config["epochs"]))
            epoch_start_time = time.time()
            total_train_losses = [0] * 4
            total_validation_losses = [0] * 4

            # Train
            self.model.train()
            for static_inputs, start_states, sim, _ in tqdm(train_loader):
                static_inputs = static_inputs.to(device, non_blocking=True)
                start_states = start_states.to(device, non_blocking=True)
                sim = sim.to(device, non_blocking=True)
                normalized_inputs = normalize_inputs(static_inputs, self.input_limits)
                normalized_start_states = normalize_start_state(start_states, self.state_limits)
                delta_labels = delta(sim)
                delta_labels = delta_labels[:, :, :9] if self.group == Group.MANIPULATOR else torch.cat((delta_labels[:, :, :2], delta_labels[:, :, -1]), dim=-1)
                normalized_delta_labels = normalize_labels(delta_labels, self.output_limits)
                losses = self._training_step(normalized_inputs, normalized_start_states, normalized_delta_labels,
                                             optimizer, loss_functions, evaluate=False)
                for i in range(len(losses)):
                    total_train_losses[i] += losses[i]
            avg_train_losses = list(map(lambda x: x / len(train_loader), total_train_losses))

            # Validate
            if len(validate_loader) > 0:
                self.model.eval()
                with torch.no_grad():
                    for static_inputs, start_states, sim, _ in tqdm(validate_loader):
                        static_inputs = static_inputs.to(device, non_blocking=True)
                        start_states = start_states.to(device, non_blocking=True)
                        sim = sim.to(device, non_blocking=True)
                        normalized_inputs = normalize_inputs(static_inputs, self.input_limits)
                        normalized_start_states = normalize_start_state(start_states, self.state_limits)
                        delta_labels = delta(sim)
                        delta_labels = delta_labels[:, :, :9] if self.group == Group.MANIPULATOR else torch.cat(
                            (delta_labels[:, :, :2], delta_labels[:, :, -1]), dim=-1)
                        normalized_delta_labels = normalize_labels(delta_labels, self.output_limits)
                        losses = self._training_step(normalized_inputs, normalized_start_states, normalized_delta_labels,
                                                     optimizer, loss_functions, evaluate=True)
                        for i in range(len(losses)):
                            total_validation_losses[i] += losses[i]
                avg_validation_losses = list(map(lambda x: x / len(validate_loader), total_validation_losses))
            else:
                avg_validation_losses = [0] * 4

            if avg_validation_losses[0] < best_validation_loss:
                best_weights = self.model.state_dict()

            training_history.append([avg_train_losses, avg_validation_losses])
            print("Avg loss: Train: {:.6f} ({:.6f} {:.6f} {:.6f}), validation: {:.6f} ({:.6f} {:.6f} {:.6f})".format(
                *avg_train_losses, *avg_validation_losses))

            print("Epoch {} took {:.2f}s".format(epoch + 1, time.time() - epoch_start_time))

            if epoch > 0 and epoch % 10 == 0:
                self._save(output_dir, training_history, epoch)

        total_training_time = time.time() - start
        print("Training took {}".format(str(datetime.timedelta(seconds=total_training_time)).split(".")[0]))
        print("Setting model weights to minimize validation loss")
        self.model.load_state_dict(best_weights)
        self._save(output_dir, training_history)

    def _training_step(self, static_inputs: torch.Tensor, points_from: torch.Tensor, delta_labels: torch.Tensor, optimizer,
                       loss_functions: dict, evaluate: bool = False):
        """
        Perform one training step
        Inputs and targets are batched!
        """
        optimizer.zero_grad()

        all_inputs = torch.cat((static_inputs, points_from), dim=-1)
        x = delta_labels.narrow(1, 0, delta_labels.size(1) - 1)  # Original sequence
        y = delta_labels.narrow(1, 1, delta_labels.size(1) - 1)  # Sequence shifted by 1 timestep

        delta_outputs_normalized, _ = self.model(all_inputs, x)

        eos_probability_loss = loss_functions["eos_probability"](delta_outputs_normalized[:, :, 0], y[:, :, 0])
        success_probability_loss = loss_functions["success_probability"](delta_outputs_normalized[:, :, 1], y[:, :, 1])
        trajectory_loss = loss_functions["trajectory"](delta_outputs_normalized[:, :, 2:], y[:, :, 2:])
        loss = trajectory_loss + eos_probability_loss + success_probability_loss

        if not evaluate:
            loss.backward()
            optimizer.step()

        return loss.item(), trajectory_loss.item(), eos_probability_loss.item(), success_probability_loss.item()

    def simulate(self, inputs_world: torch.Tensor, point_start_world: torch.Tensor, max_trajectory_len: int = 500) -> torch.Tensor:
        self.model = self.model.to(device)
        self.model.eval()

        inputs_world = inputs_world.to(device)
        point_start_world = point_start_world.to(device)
        has_batch_dim = len(inputs_world.size()) > 1
        if not has_batch_dim:
            inputs_world = inputs_world.unsqueeze(0)
            point_start_world = point_start_world.unsqueeze(0)
        batch_size = inputs_world.size(0)

        normalized_template_inputs = normalize_inputs(inputs_world, self.input_limits)
        normalized_start_state = normalize_start_state(point_start_world, self.state_limits)
        normalized_inputs = torch.cat((normalized_template_inputs, normalized_start_state), dim=-1).float()
        recurrent_inputs = torch.zeros((1, 1, self.output_size), dtype=torch.float32, device=device).repeat((batch_size, 1, 1))
        predicted_deltas_normalized = None
        hidden = None

        # Autoregressively predict trajectory
        with torch.no_grad():
            while torch.all(recurrent_inputs[:, 0, 0] < 0.5) and predicted_deltas_normalized is None \
                    or predicted_deltas_normalized.size(1) < max_trajectory_len:
                outputs, hidden = self.model(normalized_inputs, recurrent_inputs, hidden)
                if predicted_deltas_normalized is None:
                    predicted_deltas_normalized = outputs
                else:
                    predicted_deltas_normalized = torch.cat((predicted_deltas_normalized, outputs), 1)
                recurrent_inputs = outputs

        predicted_deltas_world = denormalize_outputs(predicted_deltas_normalized, self.output_limits)
        trajectory_world = undelta(predicted_deltas_world, point_start_world)

        # Build trajectory according to group mask
        batch_size = trajectory_world.size(0)
        traj_len = trajectory_world.size(1)
        if self.group == Group.MANIPULATOR:
            traj = trajectory_world
        elif self.trajectory_properties.cartesian:
            traj = torch.cat((trajectory_world[:, :, :2], point_start_world[:, :7].unsqueeze(0).repeat(1, traj_len, 1)), dim=-1)
        else:
            traj = trajectory_world[:, :, :2]
        if self.trajectory_properties.wrench:
            traj = torch.cat((traj, torch.zeros(batch_size, traj_len, 6, device=inputs_world.device)), dim=-1)
        if self.group == Group.GRIPPER:
            traj = torch.cat((traj, trajectory_world[:, :, -1].unsqueeze(-1)), dim=-1)
        elif self.trajectory_properties.gripper:
            traj = torch.cat((traj, point_start_world[:, 7].view(batch_size, 1, 1).repeat(1, traj_len, 1)), dim=-1)

        if not has_batch_dim:   # Remove batch dimension again
            traj = traj.reshape(traj_len, -1)
        return traj

    def _set_input_output_limits(self,  data: DirectoryDataset):
        if self.input_limits is None and self.output_limits is None:
            print("Setting input and output limits for this NeuralSimulator")
            loader = torch.utils.data.DataLoader(data, batch_size=256, num_workers=4)
            computed_output_limits = None
            computed_input_limits = None
            computed_state_limits = None
            for inputs, start_state, sim, real in tqdm(loader):
                sim_delta = delta(sim)
                sim_delta = sim_delta[:, :, :9] if self.group == Group.MANIPULATOR else torch.cat((sim_delta[:, :, :2], sim_delta[:, :, -1]), dim=-1)
                real_delta = delta(real)
                real_delta = real_delta[:, :, :9] if self.group == Group.MANIPULATOR else torch.cat((real_delta[:, :, :2], real_delta[:, :, -1]), dim=-1)
                output_delta = torch.cat((sim_delta, real_delta), dim=0)
                batch_output_limits = torch.stack((torch.min(output_delta.view(-1, output_delta.size(-1)), dim=0)[0],
                                                   torch.max(output_delta.view(-1, output_delta.size(-1)), dim=0)[0]))
                batch_input_limits = torch.stack((torch.min(inputs, dim=0)[0], torch.max(inputs, dim=0)[0]))
                batch_state_limits = torch.stack((torch.min(start_state, dim=0)[0], torch.max(start_state, dim=0)[0]))
                if computed_input_limits is None:
                    computed_output_limits = batch_output_limits
                    computed_input_limits = batch_input_limits
                    computed_state_limits = batch_state_limits
                else:
                    computed_output_limits = torch.stack((torch.min(batch_output_limits[0], computed_output_limits[0]),
                                                          torch.max(batch_output_limits[1], computed_output_limits[1])))
                    computed_input_limits = torch.stack((torch.min(batch_input_limits[0], computed_input_limits[0]),
                                                         torch.max(batch_input_limits[1], computed_input_limits[1])))
                    computed_state_limits = torch.stack((torch.min(batch_state_limits[0], computed_state_limits[0]),
                                                         torch.max(batch_state_limits[1], computed_state_limits[1])))

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
            for dim_idx in range(computed_state_limits.size(1)):
                if computed_state_limits[0, dim_idx] == computed_state_limits[1, dim_idx]:
                    orig_value = computed_state_limits[0, dim_idx].clone()
                    computed_state_limits[0, dim_idx] = orig_value - 1  # new min
                    computed_state_limits[1, dim_idx] = orig_value + 1  # new max

            self.input_limits = computed_input_limits
            self.state_limits = computed_state_limits
            self.output_limits = computed_output_limits


def normalize_inputs(inputs_world: torch.Tensor, input_limits: torch.Tensor) -> torch.Tensor:
    scaled_inputs = transformations.scale(inputs_world, input_limits[0], input_limits[1], -1, 1)
    # Scaling can result in NaN values due to division by zero --> set NaN to 0
    scaled_inputs[scaled_inputs != scaled_inputs] = 0
    return scaled_inputs


def normalize_start_state(start_state_world: torch.Tensor, state_limits: torch.Tensor) -> torch.Tensor:
    scaled_start_state = transformations.scale(start_state_world, state_limits[0], state_limits[1], -1, 1)
    # Scaling can result in NaN values due to division by zero --> set NaN to 0
    scaled_start_state[scaled_start_state != scaled_start_state] = 0
    return scaled_start_state


def normalize_labels(labels_world: torch.Tensor, label_limits: torch.Tensor) -> torch.Tensor:
    """
    :param labels_world: Deltas
    """
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
