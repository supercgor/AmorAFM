import logging
import numpy as np
import os
import sys
import torch
import warnings
import pandas as pd

from ase import Atoms
from multiprocessing import Pool
from pathlib import Path
from matplotlib import pyplot as plt
from functools import partial
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from torch import nn, Tensor
from torch.utils.data import Sampler
from torch.nn import functional as F
from torchmetrics import Metric
from torchvision.utils import make_grid


def get_logger(name, save_dir = None, level: int = logging.INFO):
    handler: list = [logging.StreamHandler(stream=sys.stdout)]
    if save_dir is not None:
        handler.append(logging.FileHandler(Path(save_dir) / f"log.txt"))
    
    logging.basicConfig(level=level, handlers=handler)
    
    return logging.getLogger(name)


def log_to_csv(path, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = v.detach().cpu().numpy()
            
    df = pd.DataFrame([kwargs])
    if not os.path.isfile(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)


def make_grid_centers(bbox_size, padding, rot, cell, mode = 'outer') -> np.ndarray:
    assert not np.isclose(padding, 0.0), "Padding must be non-zero!"
    bbox = np.array([[0.0,          0.0         ],
                        [bbox_size[0], 0.0         ],
                        [bbox_size[0], bbox_size[1]],
                        [0.0,          bbox_size[1]]])
    
    bbox_center = [bbox_size[0] / 2, bbox_size[1] / 2]
    
    rot_bbox = (bbox - bbox_center) @ [[np.sin(rot), np.cos(rot)], [np.cos(rot), -np.sin(rot)]] # rotate anti-clockwise in (y, x) format
    rot_bbox_size = np.max(rot_bbox, axis=0) - np.min(rot_bbox, axis=0)
    
    if mode == 'outer':
        ref_boarder = bbox_size
    elif mode == 'inner':
        ref_boarder = rot_bbox_size
    
    N = np.clip(np.rint((np.diag(cell)[:2] - ref_boarder) / padding) + 1, 1, None).astype(np.int32)
    
    if N[0] == 1:
        ipos = cell[0:1,0] / 2
    else:
        ipos = np.linspace(ref_boarder[0] / 2, cell[0,0] - ref_boarder[0] / 2, N[0])
        
    if N[1] == 1:
        jpos = cell[1:2] / 2
    else:
        jpos = np.linspace(ref_boarder[1] / 2, cell[1,1] - ref_boarder[1] / 2, N[1])
        
    return np.stack(np.meshgrid(ipos, jpos, indexing="ij"), axis=-1).reshape(-1, 2) # (N, 2)
    
def make_grid_samples_points(grid_center, bbox_size, resolution, cell, rot = 0.0):
    """
    _summary_

    Args:
        grid_center (_type_): (2, ) # (X, Y)
        bbox_size (_type_): (2, )   # (X, Y)
        resolution (_type_): (2, )  # (X, Y)
        cell (_type_): (3, 3)       # (X, Y, Z)
        rot (float): in degree
    """
    rot = np.deg2rad(rot)
    
    gridx = np.linspace(0, bbox_size[0], resolution[0])
    gridy = np.linspace(0, bbox_size[1], resolution[1])
    
    grid = np.stack(np.meshgrid(gridx, gridy, indexing = "ij"), axis = -1)
    
    bbox_center = [bbox_size[0] / 2, bbox_size[1] / 2]

    grid = (grid - bbox_center) @ [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
    grid = (grid + grid_center) @ np.linalg.inv(cell[:2, :2])
    grid = (grid - 0.5) * 2
    grid = grid[..., (1, 0)]
    return grid

class ItLoader(Sampler):
    def __init__(self, dts, iters, shuffle = False):
        self.dts = dts
        self.iters = iters
        self.shuffle = shuffle
        self.dataiter = iter(self.dts)
    
    def __iter__(self):
        if self.shuffle:
            idxs = np.random.permutation(len(self.dts))
        else:
            idxs = np.arange(len(self.dts))
        
        iters = 0
        while iters < self.iters:
            for i in idxs:
                yield i
                iters += 1
                if iters >= self.iters:
                    break
            else:
                if self.shuffle:
                    idxs = np.random.permutation(len(self.dts))
    
    def __len__(self):
        return self.iters        


class MetricCollection:
    def __init__(self, *args: Metric, **kwargs: Metric):
        if len(args) > 0:
            self.keys = list(map(str, range(len(args))))
            self._metrics = dict(zip(self.keys, args))
        elif len(kwargs) > 0:
            self.keys = list(kwargs.keys())
            self._metrics = kwargs
        else:
            raise ValueError("At least one metric should be provided")
    
    def update(self, *args, **kwargs):
        if len(args) > 0:
            for k in self.keys:
                if isinstance(args, (tuple, list)):
                    self._metrics[k].update(*args)
                else:
                    self._metrics[k].update(args)
        else:
            for k, v in kwargs.items():
                if isinstance(v, (tuple, list)):
                    self._metrics[k].update(*v)
                else:
                    self._metrics[k].update(v)
    
    def compute(self):
        return {k: v.compute() for k, v in self._metrics.items()}
    
    def __call__(self, *args, **kwargs):
        outs = {}
        if len(args) > 0:
            for k in self.keys:
                if isinstance(args, (tuple, list)):
                    v = self._metrics[k](*args)
                else:
                    v = self._metrics[k](args)
                outs[k] = v    
        else:
            for k, v in kwargs.items():
                if isinstance(v, (tuple, list)):
                    v = self._metrics[k](*v)
                else:
                    v = self._metrics[k](v)
                outs[k] = v
        return outs
        
    def reset(self):
        for v in self._metrics.values():
            v.reset()
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._metrics[self.keys[key]]
        else:
            return self._metrics[key]
    
    def to(self, device):
        for v in self._metrics.values():
            v.to(device)
        return self
    

def box2vec(box_cls, box_off, *args, threshold=0.5):
    """
    _summary_

    Args:
        box_cls (Tensor): Y X Z
        box_off (Tensor): Y X Z (ox, oy, oz)
        args (tuple[ Tensor]): Y X Z *
    Returns:
        tuple[Tensor]: (N, ), (N, 3), N
    """
    X, Y, Z = box_cls.shape
    mask = np.nonzero(box_cls > threshold)
    box_cls = box_cls[mask]
    box_off = box_off[mask] + np.stack(mask, axis=-1)
    box_off = box_off / [X, Y, Z]
    # box_off = box_off[:, [1, 0, 2]] / [Y, X, Z]
    # box_off[:, 1] = 1 - box_off[:, 1]
    args = [arg[mask] for arg in args]
    return box_cls, box_off, *args
    # Y, X, Z = box_cls.shape
    # mask = np.nonzero(box_cls > threshold)
    # box_cls = box_cls[mask]
    # box_off = box_off[mask] + np.stack([mask[1], mask[0] - Y, mask[2]], axis=-1)
    # box_off = box_off / [X, -Y, Z]
    # args = [arg[mask] for arg in args]
    # return box_cls, box_off, *args


def masknms(pos, cutoff):
    """
    _summary_

    Args:
        pos (Tensor): N 3

    Returns:
        Tensor: N 3
    """
    if isinstance(pos, Tensor):
        return _masknms_th(pos, cutoff)
    else:
        return _masknms_np(pos, cutoff)


def _masknms_np(pos, cutoff):
    mask = np.ones(pos.shape[0], dtype=np.bool_)
    for i in range(pos.shape[0]):
        if mask[i]:
            mask[i + 1:] = mask[i + 1:] & (np.sum(
                (pos[i + 1:] - pos[i])**2, axis=1) > cutoff**2)
    return mask


def _masknms_th(pos, cutoff):
    mask = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    for i in range(pos.shape[0]):
        if mask[i]:
            mask[i + 1:] = mask[i + 1:] & (torch.sum(
                (pos[i + 1:] - pos[i])**2, dim=1) > cutoff**2)
    return mask

def argmatch(pred, targ, cutoff):
    # This function is only true when one prediction does not match two targets and one target can match more than two predictions
    # return pred_ind, targ_ind
    if isinstance(pred, Tensor):
        return _argmatch_th(pred, targ, cutoff)
    else:
        return _argmatch_np(pred, targ, cutoff)


def _argmatch_np(pred: np.ndarray, targ: np.ndarray,
                 cutoff: float) -> tuple[np.ndarray, ...]:
    dis = cdist(targ, pred)
    dis = np.stack((dis < cutoff).nonzero(), axis=-1)
    dis = dis[:, (1, 0)]
    _, idx, counts = np.unique(dis[:, 1],
                               return_inverse=True,
                               return_counts=True)
    idx = np.argsort(idx)
    counts = counts.cumsum()
    if counts.shape[0] != 0:
        counts = np.concatenate(([0], counts[:-1]))
    idx = idx[counts]
    dis = dis[idx]
    return dis[:, 0], dis[:, 1]


def _argmatch_th(pred: Tensor, targ: Tensor,
                 cutoff: float) -> tuple[Tensor, ...]:
    dis = torch.cdist(targ, pred)
    dis = (dis < cutoff).nonzero()
    dis = dis[:, (1, 0)]
    _, idx, counts = torch.unique(dis[:, 1],
                                  sorted=True,
                                  return_inverse=True,
                                  return_counts=True)
    idx = torch.argsort(idx, stable=True)
    counts = counts.cumsum(0)
    if counts.shape[0] != 0:
        counts = torch.cat([
            torch.as_tensor([0], dtype=counts.dtype, device=counts.device),
            counts[:-1]
        ])
    idx = idx[counts]
    dis = dis[idx]
    return dis[:, 0], dis[:, 1]


def group_as_water(O_position, H_position):
    """
    Group the oxygen and hydrogen to water molecule

    Args:
        pos_o (ndarray | Tensor): shape (N, 3)
        pos_h (ndarray | Tensor): shape (2N, 3)

    Returns:
        ndarray | Tensor: (N, 9)
    """
    if isinstance(O_position, Tensor):
        dis = torch.cdist(O_position, H_position)
        dis = torch.topk(dis, 2, dim=1, largest=False).indices
        return torch.cat([O_position, H_position[dis].view(-1, 6)], dim=-1)
    else:
        dis = cdist(O_position, H_position)
        dis = np.argpartition(dis, 1, axis=1)[:, :2]
        return np.concatenate([O_position, H_position[dis].reshape(-1, 6)],
                              axis=-1)

def box2orgvec(box,
               threshold=0.5,
               cutoff=2.0,
               real_size=(25, 25, 12),
               sort=True,
               nms=True):
    """
    Convert the prediction/target to the original vector, including the confidence sequence, position sequence, and rotation matrix sequence

    Args:
        box (Tensor): X Y Z 10
        threshold (float): confidence threshold
        cutoff (float): nms cutoff distance
        real_size (Tensor): real size of the box
        sort (bool): to sort the box by confidence
        nms (bool): to nms the box

    Returns:
        tuple[Tensor]: `conf (N,)`, `pos (N, 3)`, `R (N, 3, 3)`
    """
    if box.shape[-1] == 4:
        pd_conf, pd_pos, *_ = box2vec(box[..., 0], box[..., 1:4], threshold=threshold)
        pd_pos = pd_pos * real_size
        if sort:
            pd_conf_order = pd_conf.argsort()[::-1]
            pd_pos = pd_pos[pd_conf_order]
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
            
        return pd_conf, pd_pos

    else:
        raise ValueError(
            f"Require the last dimension of the box to be 4 or 10, but got {box.shape[-1]}"
        )

# def vec2box(unit_pos, vec=None, box_size=(25, 25, 12)):
#     vec_dim = 0 if vec is None else vec.shape[-1]
#     box = np.zeros((box_size[1], box_size[0], box_size[2], 4 + vec_dim))
#     unit_pos = np.clip(unit_pos, 0, 1 - 1E-7)
#     unit_pos[:, 1] = 1 - unit_pos[:, 1]
#     pd_ind = np.floor(unit_pos * box_size).astype(np.int_)
#     pd_off = unit_pos * box_size - pd_ind
#     if vec is None:
#         feature = np.concatenate([np.ones((unit_pos.shape[0], 1)), pd_off],
#                                     axis=-1)
#     else:
#         feature = np.concatenate(
#             [np.ones((unit_pos.shape[0], 1)), pd_off, vec], axis=-1)
#     box[pd_ind[:, 1], pd_ind[:, 0], pd_ind[:, 2]] = feature
#     return box

def vec2box(unit_pos, vec=None, box_size=(25, 25, 12)):
    vec_dim = 0 if vec is None else vec.shape[-1]
    box = np.zeros((box_size[0], box_size[1], box_size[2], 4 + vec_dim))
    unit_pos = np.clip(unit_pos, 0, 1 - 1E-7)
    pd_ind = np.floor(unit_pos * box_size).astype(np.int_)
    pd_off = unit_pos * box_size - pd_ind
    if vec is None:
        feature = np.concatenate([np.ones((unit_pos.shape[0], 1)), pd_off], axis=-1)
    else:
        feature = np.concatenate([np.ones((unit_pos.shape[0], 1)), pd_off, vec], axis=-1)
    box[pd_ind[:, 0], pd_ind[:, 1], pd_ind[:, 2]] = feature
    return box

def box2atom(box,
             cell=[25.0, 25.0, 16.0],
             threshold=0.5,
             cutoff: float | tuple[float, float] = 2.0,
             nms=True,
             num_workers = 1,
             order = ("O", "H"),
             ):
    
    if box.ndim > 4:
        if num_workers > 1:
            fn = partial(box2atom, cell=cell, threshold=threshold, cutoff=cutoff, nms=nms, order = order)
            with Pool() as p:
                return p.map(fn, box)
        else:
            return list(map(lambda x: box2atom(x, cell=cell, threshold=threshold, cutoff=cutoff, nms=nms, order = order), box))

    atoms = Atoms(cell=cell, pbc=False)
    for i in range(box.shape[-1] // 4):
        cut = cutoff[0] if isinstance(cutoff, tuple) else cutoff
        conf, pos = box2orgvec(box[..., i * 4:i * 4 + 4], threshold, cut, cell, sort=True, nms=nms)
        new_atoms = Atoms(order[i] * pos.shape[0], pos)
        new_atoms.set_array('conf', conf)
        atoms += new_atoms

    return atoms


def makewater(pos: np.ndarray, rot: np.ndarray):
    # N 3, N 3 3 -> N 3 3
    if not isinstance(pos, np.ndarray):
        pos = pos.detach().cpu().numpy()
    if not isinstance(rot, np.ndarray):
        rot = rot.detach().cpu().numpy()

    water = np.array([
        [0., 0., 0.],
        [0., 0., 0.9572],
        [0.9266272, 0., -0.23998721],
    ])

    # print( np.einsum("ij,Njk->Nik", water, rot) )
    return np.einsum("ij,Njk->Nik", water, rot) + pos[:, None, :]


def __encode_th(positions):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[..., 0:3], positions[..., 3:6], positions[..., 6:]
    u, v = u + v - 2 * o, u - v
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = v / torch.norm(v, dim=-1, keepdim=True)
    v = torch.where(v[..., 1].unsqueeze(-1) >= 0, v, -v)
    v = torch.where(v[..., 0].unsqueeze(-1) >= 0, v, -v)
    return torch.cat([o, u, v], dim=-1)


def __encode_np(positions: np.ndarray):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[..., 0:3], positions[..., 3:6], positions[..., 6:]
    u, v = u + v - 2 * o, u - v
    u = u / np.expand_dims(((u**2).sum(axis=-1)**0.5), -1)
    v = v / np.expand_dims(((v**2).sum(axis=-1)**0.5), -1)
    v = np.where(v[..., 1][..., None] >= 0, v, -v)
    v = np.where(v[..., 0][..., None] >= 0, v, -v)
    return np.concatenate((o, u, v), axis=-1)


def __decode_th(emb):
    o, u, v = emb[..., 0:3], emb[..., 3:6], emb[..., 6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return torch.cat([o, h1, h2], dim=-1)


def __decode_np(emb):
    o, u, v = emb[..., 0:3], emb[..., 3:6], emb[..., 6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return np.concatenate((o, h1, h2), axis=-1)


def encodewater(positions):
    if isinstance(positions, torch.Tensor):
        return __encode_th(positions)
    else:
        return __encode_np(positions)


def decodewater(emb):
    if isinstance(emb, Tensor):
        return __decode_th(emb)
    else:
        return __decode_np(emb)


def rotate(points, rotation_vector: np.ndarray):
    """
    Rotate the points with rotation_vector.

    Args:
        points (_type_): shape (..., 3)
        rotation_vector (_type_): rotation along x, y, z axis. shape (3,)
    """
    if points.shape[-1] == 2:
        rotation_vector = np.array([0, 0, rotation_vector])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    if isinstance(points, Tensor):
        rotation_matrix = torch.as_tensor(rotation_matrix)

    if points.shape[-1] == 3:
        return points @ rotation_matrix.T

    elif points.shape[-1] == 2:
        return points @ rotation_matrix.T[:2, :2]

    else:
        raise NotImplementedError("Only 2D and 3D rotation is implemented.")


def logit(x, eps=1E-7):
    if isinstance(x, (float, np.ndarray)):
        return -np.log(1 / (x + eps) - 1)
    else:
        return torch.logit(x, eps)


def replicate(points: np.ndarray, times: list[int], offset: np.ndarray) -> np.ndarray:
    """
    Replicate the points with times and offset.

    Args:
        points (ndarray): shape (N, 3)
        times (list[int]): [x times, y times, z times]
        offset (ndarray): shape (3, 3) 3 vectors

    Returns:
        _type_: _description_
    """
    if len(offset.shape) == 1:
        offset = np.diag(offset)

    for i, (t, o) in enumerate(zip(times, offset)):
        if t == 1:
            continue
        buf = []
        low = -(t // 2)
        for j in range(low, low + t):
            res = points + j * o
            buf.append(res)
        points = np.concatenate(buf, axis=0)
    return points


def combine_atoms(atom_list, eps=0.5):
    all_atoms = atom_list[0]
    for atoms in atom_list[1:]:
        all_atoms = all_atoms + atoms

    o_atoms = all_atoms[all_atoms.symbols == 'O']
    h_atoms = all_atoms[all_atoms.symbols == 'H']

    o_cluster = DBSCAN(eps=eps, min_samples=1).fit(o_atoms.positions)
    h_cluster = DBSCAN(eps=0.25, min_samples=1).fit(h_atoms.positions)
    o_tag = np.zeros(len(o_cluster.components_))
    o_tag_count = np.zeros(len(o_cluster.components_))
    np.add.at(o_tag, o_cluster.labels_, o_atoms.get_tags())
    np.add.at(o_tag_count, o_cluster.labels_, 1)

    h_tag = np.zeros(len(h_cluster.components_))
    h_tag_count = np.zeros(len(h_cluster.components_))
    np.add.at(h_tag, h_cluster.labels_, h_atoms.get_tags())
    np.add.at(h_tag_count, h_cluster.labels_, 1)

    o_atoms = Atoms(symbols='O',
                    positions=o_cluster.components_,
                    tags=o_tag / o_tag_count)
    h_atoms = Atoms(symbols='H',
                    positions=h_cluster.components_,
                    tags=h_tag / h_tag_count)

    out = o_atoms + h_atoms

    return out


def plot_preditions(save_path, afms, pred_atoms, atoms, name = None):
    # afms: C Z H W
    cell = atoms.cell.array
    X, Y, Z = np.diag(cell)
    
    tgt_Opos = atoms[atoms.numbers == 8].positions
    tgt_Hpos = atoms[atoms.numbers == 1].positions
    
    pred_positions = pred_atoms.positions
    pred_numbers = pred_atoms.numbers
    pred_color = np.where(pred_numbers == 8, 'b', 'lightgray')
    
    order = np.argsort(pred_positions[:, 2])
    pred_positions = pred_positions[order]
    pred_numbers = pred_numbers[order]
    
    afms = np.rot90(afms)
    
    fig = plt.figure(figsize=(12, 4))
    if name is not None:
        fig.suptitle(name)
    axs = fig.subplots(1, 3, sharey=True)

    for i, idx in enumerate([0, 4, 7]):
        axs[i].imshow(afms[idx].transpose((1, 2, 0)), extent=[0, X, 0, Y])
        axs[i].scatter(tgt_Opos[:, 0], tgt_Opos[:, 1], c='r', marker='o')
        axs[i].scatter(tgt_Hpos[:, 0], tgt_Hpos[:, 1], c='w', marker='o')
        axs[i].scatter(pred_positions[:, 0], pred_positions[:, 1], c = pred_color, marker='x')
        axs[i].set_xlim([0, X])
        axs[i].set_ylim([0, Y])
    
    plt.savefig(save_path)
    plt.close()
    
def write_water_data(atoms, 
                     fname, 
                     padding = [25, 25, 10], 
                     ndx: dict[str, set | list | tuple] | None = None, 
                     atol = 0.1
                     ):
    cell = np.diag(atoms.cell)
    O_idxs = np.where(atoms.numbers == 8)[0]
    H_idxs = np.where(atoms.numbers == 1)[0]
    O_atoms = atoms[O_idxs]
    H_atoms = atoms[H_idxs]
    H_tree = KDTree(H_atoms.positions)
    map_ndx = {}
    part1 = f"""
{len(atoms)} atoms
{len(H_atoms)} bonds
{len(O_atoms)} angles
2 atom types
1 bond types
1 angle types
-{padding[0]:.1f} {cell[0] + padding[0]:.1f} xlo xhi
-{padding[1]:.1f} {cell[1] + padding[1]:.1f} ylo yhi
-{padding[2]:.1f} {cell[2] + padding[2]:.1f} zlo zhi

Masses

1 15.9994
2 1.008

Pair Coeffs

1 0.21084 3.1668
2 0 0

Bond Coeffs

1 10000 0.9572

Angle Coeffs

1 10000 104.52

Atoms

"""
    part2 = ""
    
    a_num = 1
    for i, (o_idx, pos) in enumerate(zip(O_idxs, O_atoms.positions)):
        str_pos = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        part2 += f"{a_num} {i} 1 -1.1794 {str_pos} 0 0 0\n"
        if ndx is not None:
            for key, val in ndx.items():
                if o_idx in val:
                    if key in map_ndx:
                        map_ndx[key].append(a_num)
                    else:
                        map_ndx[key] = [a_num]
                    
        a_num += 1
        h_idxs = H_tree.query_ball_point(pos, 1.0, p=2)
        okays = 0
        for i, h_sidx in enumerate(h_idxs):
            if okays >=2:
                break
            
            hpos = np.linalg.norm(H_atoms.positions[h_sidx] - pos)

            if np.isclose(hpos, 0.9547, atol = atol, rtol=0.03):
                okays += 1
                str_pos = f"{H_atoms.positions[h_sidx][0]:.4f} {H_atoms.positions[h_sidx][1]:.4f} {H_atoms.positions[h_sidx][2]:.4f}"
                part2 += f"{a_num} {i} 2 0.5897 {str_pos} 0 0 0\n"
                if ndx is not None:
                    for key, val in ndx.items():
                        h_idx = H_idxs[h_sidx]
                        if h_idx in val:
                            if key in map_ndx:
                                map_ndx[key].append(a_num)
                            else:
                                map_ndx[key] = [a_num]
                a_num += 1
            else:
                if len(h_idxs[i+1:]) < 2 - okays:
                    warnings.warn(f"Hydrogen bond length is not close to 0.9546: {hpos}")
                
    a_num -= 1
    part3 = """
Bonds

"""
        
    for i in range(len(H_atoms)):
        part3 += f"{i + 1} 1 {1 + (i//2 * 3)} {(i//2) * 3 + 2 + i % 2}\n"
        
    part4 = """
Angles

"""
        
    for i in range(len(O_atoms)):
        part4 += f"{i+1} 1 {i*3 + 2} {i*3 +1} {i*3 + 3}\n"
    
    all_str = part1 + part2 + part3 + part4
    
    with open(fname, 'w') as f:
        f.write(all_str)
        
    if ndx is not None:
        fname = os.path.splitext(fname)[0] + ".ndx"
        with open(fname, 'w') as f:
            for key, val in map_ndx.items():
                f.write(f"[ {key} ]\n")
                val = list(map(lambda x: f"{x:4d}", val))
                for i in range(0, len(val), 15):
                    f.write(" ".join(val[i:i+15]) + "\n")


class ConfusionMatrix(Metric):
    def __init__(self, 
                 count_types: tuple[str, ...] = ("O", "H"),
                 real_size: tuple[float, ...] = (25.0, 25.0, 3.0), 
                 match_distance: float = 1.0, 
                 split: list[float] = [0.0, 3.0]
                 ):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        # TP, FP, FN, AP, AR, F1, SUC
        self.add_state("matrix", default = torch.zeros((len(count_types), len(split) - 1, 7), dtype = torch.float), dist_reduce_fx = "sum")
        self.add_state("total", default = torch.tensor([0]), dist_reduce_fx = "sum")
        
        self.matrix: torch.Tensor
        self.total: torch.Tensor
        
        self.count_types = count_types
        self.match_distance = match_distance
        self.split = split
        self.split[-1] += 1e-5
    
    def update(self, preds: list[Atoms], targs: list[Atoms]):
        if isinstance(preds, Atoms):
            preds = [preds]
        if isinstance(targs, Atoms):
            targs = [targs]
            
        for b, (pred, targ) in enumerate(zip(preds, targs)):
            for t, count_type in enumerate(self.count_types):
                pre = pred[pred.symbols == count_type].positions # type: ignore
                tar = targ[targ.symbols == count_type].positions # type: ignore
                
                pd_match_ids, tg_match_ids = argmatch(pre, tar, self.match_distance)
                
                match_pd_pos = pre[pd_match_ids] # N 3
                match_tg_pos = tar[tg_match_ids] # N 3
                
                mask = np.isin(range(len(pre)), pd_match_ids, invert=True)
                non_match_pd_pos = pre[mask] # N 3
                mask = np.isin(range(len(tar)), tg_match_ids, invert=True)
                non_match_tg_pos = tar[mask]
                
                
                for i, (low, high) in enumerate(zip(self.split[:-1], self.split[1:])):
                    num_non_match_pd = ((non_match_pd_pos[:, 2] >= low) & (non_match_pd_pos[:, 2] < high)).sum() # FP
                    num_non_match_tg = ((non_match_tg_pos[:, 2] >= low) & (non_match_tg_pos[:, 2] < high)).sum() # FN
                    num_match = ((match_tg_pos[:, 2] >= low) & (match_tg_pos[:, 2] < high)).sum() # TP
                    self.matrix[t, i, 0] += num_match # TP
                    self.matrix[t, i, 1] += num_non_match_pd # FP
                    self.matrix[t, i, 2] += num_non_match_tg # FN
                    self.matrix[t, i, 3] += num_match / (num_non_match_pd + num_match) if num_match > 0 else 1 if num_non_match_pd == 0 else 0 # AP
                    self.matrix[t, i, 4] += num_match / (num_non_match_tg + num_match) if num_match > 0 else 1 if num_non_match_tg == 0 else 0 # AR
                    self.matrix[t, i, 5] += 2 * num_match / (num_non_match_pd + num_non_match_tg + 2 * num_match) if num_match > 0 else 1 if (num_non_match_pd + num_non_match_tg) == 0 else 0 # F1
                    self.matrix[t, i, 6] += (num_non_match_pd == 0) & (num_non_match_tg == 0) # SUC
                
            self.total += 1
                    
    def compute(self):
        out = self.matrix.clone()
        out[:, :, 3:] = out[:, :, 3:] / self.total
        return out
    
class MeanAbsoluteDistance(Metric):
    def __init__(self, 
                 count_types: tuple[str, ...] = ("O", "H"),
                 real_size: tuple[float, ...] = (25.0, 25.0, 3.0), 
                 match_distance: float = 1.0, 
                 split: list[float] = [0.0, 3.0]
                 ):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        # TP, FP, FN, AP, AR, ACC, SUC
        self.add_state("RDE", default = torch.zeros(len(count_types), len(split) - 1), dist_reduce_fx = "sum")
        self.add_state("TOT", default = torch.zeros(len(count_types), len(split) - 1), dist_reduce_fx = "sum")
        
        self.RDE: torch.Tensor
        self.TOT: torch.Tensor
        
        self.count_types = count_types
        self.match_distance = match_distance
        self.split = split
        self.split[-1] += 1e-5
    
    def update(self, preds: list[Atoms], targs: list[Atoms]):
        if isinstance(preds, Atoms):
            preds = [preds]
        if isinstance(targs, Atoms):
            targs = [targs]
            
        for b, (pred, targ) in enumerate(zip(preds, targs)):
            for t, count_type in enumerate(self.count_types):
                pre = pred[pred.symbols == count_type].positions # type: ignore
                tar = targ[targ.symbols == count_type].positions # type: ignore
                
                pd_match_ids, tg_match_ids = argmatch(pre, tar, self.match_distance)
                
                match_pd_pos = pre[pd_match_ids] # N 3
                match_tg_pos = tar[tg_match_ids] # N 3
                
                for i, (low, high) in enumerate(zip(self.split[:-1], self.split[1:])):
                    mask = (match_tg_pos[:, 2] >= low) & (match_tg_pos[:, 2] < high)
                    self.RDE[t, i] += np.linalg.norm(match_pd_pos[mask] - match_tg_pos[mask], axis = 1).sum()
                    self.TOT[t, i] += mask.sum()

    def compute(self):
        return self.RDE / self.TOT
    
class MeanSquareDistance(Metric):
    def __init__(self, 
                 count_types: tuple[str, ...] = ("O", "H"),
                 real_size: tuple[float, ...] = (25.0, 25.0, 3.0), 
                 match_distance: float = 1.0, 
                 split: list[float] = [0.0, 3.0]
                 ):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        # TP, FP, FN, AP, AR, ACC, SUC
        self.add_state("RDE", default = torch.zeros(len(count_types), len(split) - 1), dist_reduce_fx = "sum")
        self.add_state("TOT", default = torch.zeros(len(count_types), len(split) - 1), dist_reduce_fx = "sum")
        
        self.RDE: torch.Tensor
        self.TOT: torch.Tensor
        
        self.count_types = count_types
        self.match_distance = match_distance
        self.split = split
        self.split[-1] += 1e-5
    
    def update(self, preds: list[Atoms], targs: list[Atoms]):
        if isinstance(preds, Atoms):
            preds = [preds]
        if isinstance(targs, Atoms):
            targs = [targs]
            
        for b, (pred, targ) in enumerate(zip(preds, targs)):
            for t, count_type in enumerate(self.count_types):
                pre = pred[pred.symbols == count_type].positions # type: ignore
                tar = targ[targ.symbols == count_type].positions # type: ignore
                
                pd_match_ids, tg_match_ids = argmatch(pre, tar, self.match_distance)
                
                match_pd_pos = pre[pd_match_ids] # N 3
                match_tg_pos = tar[tg_match_ids] # N 3
                
                for i, (low, high) in enumerate(zip(self.split[:-1], self.split[1:])):
                    mask = (match_tg_pos[:, 2] >= low) & (match_tg_pos[:, 2] < high)
                    self.RDE[t, i] += np.square(np.linalg.norm(match_pd_pos[mask] - match_tg_pos[mask], axis = 1)).sum()
                    self.TOT[t, i] += mask.sum()

    def compute(self):
        return self.RDE / self.TOT
    
class ValidProbability(Metric):
    def __init__(self, 
                 real_size: tuple[float, ...] = (25.0, 25.0, 3.0), 
                 match_distance: float = 1.1, 
                 split: list[float] = [0.0, 3.0]
                 ):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        # TP, FP, FN, AP, AR, ACC, SUC
        self.add_state("VAL", default = torch.zeros(2, len(split) - 1), dist_reduce_fx = "sum")
        self.add_state("TOT", default = torch.zeros(2, len(split) - 1), dist_reduce_fx = "sum")
        
        self.VAL: torch.Tensor
        self.TOT: torch.Tensor
        
        self.match_distance = match_distance
        self.split = split
        self.split[-1] += 1e-5
    
    def update(self, atoms: list[Atoms]):
        if isinstance(atoms, Atoms):
            atoms = [atoms]

            
        for b, ats in enumerate(atoms):
            h_pos = ats.positions[ats.symbols == "H"]
            o_pos = ats.positions[ats.symbols == "O"]
            
            htree = KDTree(h_pos)
            otree = KDTree(o_pos)
            
            o_neighbors = htree.query_ball_tree(otree, r = self.match_distance)
            h_neighbors = otree.query_ball_tree(htree, r = self.match_distance)
            
            o_val = np.array([len(neigh) == 2 for neigh in h_neighbors])
            h_val = np.array([len(neigh) == 1 for neigh in o_neighbors])
            # print(o_val, h_val)
            
            for i, (low, high) in enumerate(zip(self.split[:-1], self.split[1:])):
                mask = (o_pos[:, 2] >= low) & (o_pos[:, 2] < high)
                
                self.VAL[0, i] += o_val[mask].sum()
                self.TOT[0, i] += mask.sum()
                
                mask = (h_pos[:, 2] >= low) & (h_pos[:, 2] < high)
                self.VAL[1, i] += h_val[mask].sum()
                self.TOT[1, i] += mask.sum()
            
            
    def compute(self):
        return self.VAL / self.TOT
    

def water_solver(atoms, r = 0.9572, deg = 104.52):
    from scipy.spatial import KDTree
    deg = np.deg2rad(deg)
    occupy = set()
    o_nei = {}
    
    o_mask = atoms.numbers == 8
    o_l2g = o_mask.nonzero()[0]
    
    h_mask = atoms.numbers == 1
    h_l2g = h_mask.nonzero()[0]
    
    tree = KDTree(atoms.positions[h_mask])
    
    results = tree.query_ball_point(atoms.positions[o_mask], r = 2.0, p =2)
    
    for o_l_i, res in enumerate(results):
        o_g_i = o_l2g[o_l_i]
        o_nei[o_g_i] = []
        for h_l_i in res:
            h_g_i = h_l2g[h_l_i]
            if h_g_i not in occupy and len(o_nei[o_g_i]) < 2:
                occupy.add(h_g_i)
                o_nei[o_g_i].append(h_g_i)
            else:
                continue

    h_atoms = []
    for o, h in o_nei.items():
        o_pos = atoms.positions[o]
        if len(h) == 0:
            h1 = o_pos + np.random.randn(3)
            h2 = None
        elif len(h) == 1:
            h1 = atoms.positions[h[0]]
            h2 = None
        else:
            h1 = atoms.positions[h[0]]
            h2 = atoms.positions[h[1]]
        
        if h2 is None:
            dh1 = h1 - o_pos
            dh2 = np.random.randn(3)
            
            dh1 = dh1 / np.linalg.norm(dh1)
            dh2 = dh2 - np.dot(dh1, dh2) * dh1
            dh2 = dh2 / np.linalg.norm(dh2)
            
            dh2 = dh2 * np.sin(deg) + dh1 * np.cos(deg)
            
            h1 = o_pos + dh1 * r
            h2 = o_pos + dh2 * r
            
        else:
            dh1 = h1 - o_pos
            dh2 = h2 - o_pos
            
            dh1 = dh1 / np.linalg.norm(dh1)
            dh2 = dh2 / np.linalg.norm(dh2)
            
            dh1, dh2 = dh1 + dh2, dh1 - dh2
            
            dh1 = dh1 / np.linalg.norm(dh1)
            dh2 = dh2 / np.linalg.norm(dh2)
            
            h1 = o_pos + (dh1 * np.cos(deg / 2) + dh2 * np.sin(deg / 2)) * r
            h2 = o_pos + (dh1 * np.cos(deg / 2) - dh2 * np.sin(deg / 2)) * r
        
        h_atoms.append(h1)
        h_atoms.append(h2)
        
    o_atoms = atoms[o_mask]
    
    h_atoms = Atoms("H" * len(h_atoms), positions = h_atoms)
    
    return o_atoms + h_atoms