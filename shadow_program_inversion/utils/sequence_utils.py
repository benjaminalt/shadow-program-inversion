"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import torch


def unpad_padded_sequence(seq: torch.Tensor):
    if len(seq.size()) > 2:
        raise ValueError("pad_padded_sequence: Don't expect batched tensors")
    end_idx = torch.cumsum(seq[:, 0] < 0.5, 0)[-1]
    return seq[:end_idx.int().item() + 1]


def pad_padded_sequence(seq: torch.Tensor, length: int):
    if len(seq.size()) > 2:
        raise ValueError("pad_padded_sequence: Don't expect batched tensors")
    orig_length = seq.size(0)
    if orig_length < length:
        num_paddings = length - orig_length
        padding = seq[-1,:].unsqueeze(0).repeat(num_paddings, 1)
        padded_seq = torch.cat((seq, padding), dim=0)
        padded_seq[orig_length:, 0] = 1.0
        return padded_seq
    cut_off_seq = seq[:length]
    cut_off_seq[-1, 0] = 1.0
    return cut_off_seq


def delta(traj_batch: torch.Tensor) -> torch.Tensor:
    metadata = traj_batch[:, :, :2]
    trajectories = traj_batch[:,:,2:]
    deltas = trajectories[:,1:] - trajectories[:,:-1]
    return torch.cat((metadata[:,:-1], deltas),  dim=-1)


def undelta(delta_batch: torch.Tensor, point_start: torch.Tensor) -> torch.Tensor:
    metadata = delta_batch[:,:,:2]
    deltas = delta_batch[:,:-1,2:]
    start_and_deltas = torch.cat((point_start.unsqueeze(1), deltas), dim=1)
    return torch.cat((metadata, start_and_deltas.cumsum(dim=1)), dim=-1)


def chunk(l, n):
    """
    Yield successive n-sized chunks from l.
    See https://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]