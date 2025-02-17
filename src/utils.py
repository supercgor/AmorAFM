from ase import Atoms
from sklearn.cluster import DBSCAN
import numpy as np
from typing import overload
from numpy import ndarray
from torch import Tensor
from multiprocessing import Pool
import warnings
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.nn import functional as F

import os
import warnings
from ase.atoms import Atoms, Atom
from ase import io

import numpy as np
from scipy.spatial import KDTree
from h5py import File

def box2vec(box_cls, box_off, *args, threshold=0.5):
    """
    _summary_

    Args:
        box_cls (Tensor): X Y Z
        box_off (Tensor): X Y Z (ox, oy, oz)
        args (tuple[ Tensor]): X Y Z *
    Returns:
        tuple[Tensor]: (N, ), (N, 3), N
    """
    if isinstance(box_cls, Tensor):
        return _box2vec_th(box_cls, box_off, *args, threshold=threshold)
    elif isinstance(box_cls, np.ndarray):
        return _box2vec_np(box_cls, box_off, *args, threshold=threshold)


def _box2vec_np(box_cls, box_off, *args, threshold=0.5):
    box_size = box_cls.shape
    mask = np.nonzero(box_cls > threshold)
    box_cls = box_cls[mask]
    box_off = box_off[mask] + np.stack(mask, axis=-1)
    box_off = box_off / box_size
    args = [arg[mask] for arg in args]
    return box_cls, box_off, *args


def _box2vec_th(box_cls, box_off, *args, threshold=0.5):
    box_size = box_cls.shape
    mask = torch.nonzero(box_cls > threshold, as_tuple=True)
    box_cls = box_cls[mask]
    box_off = box_off[mask] + torch.stack(mask, dim=-1)
    box_off = box_off / torch.as_tensor(
        box_size, dtype=box_cls.dtype, device=box_cls.device)
    args = [arg[mask] for arg in args]
    return box_cls, box_off, *args


@overload
def masknms(pos: ndarray, cutoff: float) -> ndarray:
    ...


@overload
def masknms(pos: Tensor, cutoff: float) -> Tensor:
    ...


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


# def _masknms_np(pos: ndarray, cutoff: float) -> ndarray:
#     dis = cdist(pos, pos) < cutoff
#     dis = np.triu(dis, 1).astype(float)
#     args = np.ones(pos.shape[0], dtype = bool)
#     while True:
#         N = pos.shape[0]
#         restrain_tensor = dis.sum(0)
#         restrain_tensor -= (restrain_tensor != 0).astype(float) @ dis
#         SELECT = restrain_tensor == 0
#         dis = dis[SELECT][:, SELECT]
#         pos = pos[SELECT]
#         args[args.nonzero()] = SELECT
#         if N == pos.shape[0]:
#             break
#     return args

# def _masknms_th(pos: Tensor, cutoff: float) -> Tensor:
#     dis = torch.cdist(pos, pos) < cutoff
#     dis = torch.triu(dis, 1).float()
#     args = torch.ones(pos.shape[0], dtype = torch.bool, device = pos.device)
#     while True:
#         N = pos.shape[0]
#         restrain_tensor = dis.sum(0)
#         restrain_tensor -= ((restrain_tensor != 0).float() @ dis)
#         SELECT = restrain_tensor == 0
#         dis = dis[SELECT][:, SELECT]
#         pos = pos[SELECT]
#         args[args.nonzero(as_tuple=True)] = SELECT
#         if N == pos.shape[0]:
#             break
#     return args

# @overload
# def argmatch(pred: ndarray, targ: ndarray, cutoff: float) -> tuple[ndarray, ...]: ...
# @overload
# def argmatch(pred: Tensor, targ: Tensor, cutoff: float) -> tuple[Tensor, ...]: ...


def argmatch(pred, targ, cutoff):
    # This function is only true when one prediction does not match two targets and one target can match more than two predictions
    # return pred_ind, targ_ind
    if isinstance(pred, Tensor):
        return _argmatch_th(pred, targ, cutoff)
    else:
        return _argmatch_np(pred, targ, cutoff)


def _argmatch_np(pred: ndarray, targ: ndarray,
                 cutoff: float) -> tuple[ndarray, ...]:
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
    if isinstance(box, Tensor):
        return _box2orgvec_th(box, threshold, cutoff, real_size, sort, nms)
    elif isinstance(box, np.ndarray):
        return _box2orgvec_np(box, threshold, cutoff, real_size, sort, nms)


def _box2orgvec_np(box, threshold, cutoff, real_size, sort,
                   nms) -> tuple[ndarray, ...]:
    if box.shape[-1] == 4:
        pd_conf, pd_pos = box2vec(box[..., 0:1],
                                  box[..., 1:4],
                                  threshold=threshold)
        pd_pos = pd_pos * real_size
        if sort:
            pd_conf_order = pd_conf.argsort()[::-1]
            pd_pos = pd_pos[pd_conf_order]
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
        return pd_conf, pd_pos

    elif box.shape[-1] == 10:
        pd_conf, pd_pos, pd_rotx, pd_roty = box2vec(box[..., 0],
                                                    box[..., 1:4],
                                                    box[..., 4:7],
                                                    box[..., 7:10],
                                                    threshold=threshold)
        pd_rotz = np.cross(pd_rotx, pd_roty)
        pd_R = np.stack([pd_rotx, pd_roty, pd_rotz], axis=-2)
        if sort:
            pd_conf_order = pd_conf.argsort()[::-1]
            pd_pos = pd_pos[pd_conf_order]
            pd_R = pd_R[pd_conf_order]
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
            pd_R = pd_R[pd_nms_mask]
        return pd_conf, pd_pos, pd_R

    else:
        raise ValueError(
            f"Require the last dimension of the box to be 4 or 10, but got {box.shape[-1]}"
        )


def _box2orgvec_th(box, threshold, cutoff, real_size, sort, nms):
    if box.shape[-1] == 4:
        pd_conf, pd_pos = box2vec(box[..., 0],
                                  box[..., 1:4],
                                  threshold=threshold)
        pd_pos = pd_pos * torch.as_tensor(
            real_size, dtype=pd_pos.dtype, device=pd_pos.device)
        if sort:
            pd_conf_order = pd_conf.argsort(descending=True)
            pd_pos = pd_pos[pd_conf_order]

        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]

        return pd_conf, pd_pos

    elif box.shape[-1] == 10:
        pd_conf, pd_pos, pd_rotx, pd_roty = box2vec(box[..., 0],
                                                    box[..., 1:4],
                                                    box[..., 4:7],
                                                    box[..., 7:10],
                                                    threshold=threshold)
        pd_pos = pd_pos * torch.as_tensor(
            real_size, dtype=pd_pos.dtype, device=pd_pos.device)
        pd_rotz = torch.cross(pd_rotx, pd_roty, dim=-1)
        pd_R = torch.stack([pd_rotx, pd_roty, pd_rotz], dim=-2)

        if sort:
            pd_conf_order = pd_conf.argsort(descending=True)
            pd_pos = pd_pos[pd_conf_order]
            pd_R = pd_R[pd_conf_order]

        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
            pd_R = pd_R[pd_nms_mask]

        return pd_conf, pd_pos, pd_R

    else:
        raise ValueError(
            f"Require the last dimension of the box to be 4 or 10, but got {box.shape[-1]}"
        )


@overload
def vec2box(unit_pos: ndarray, vec: ndarray | None,
            box_size: tuple[int, int, int]) -> ndarray:
    ...


@overload
def vec2box(unit_pos: Tensor, vec: Tensor | None,
            box_size: tuple[int, int, int]) -> Tensor:
    ...


def vec2box(unit_pos, vec=None, box_size=(25, 25, 12)):
    vec_dim = 0 if vec is None else vec.shape[-1]
    if isinstance(unit_pos, Tensor):
        box = torch.zeros((*box_size, 4 + vec_dim),
                          dtype=unit_pos.dtype,
                          device=unit_pos.device)
        box_size = torch.as_tensor(box_size,
                                   dtype=unit_pos.dtype,
                                   device=unit_pos.device)
        pd_ind = torch.floor(unit_pos.clamp(0, 1 - 1E-7) *
                             box_size[None]).long()
        all_same = ((pd_ind[None] - pd_ind[:, None]) == 0).all(dim=-1)
        all_same.fill_diagonal_(False)
        if all_same.any():
            warnings.warn(
                f"There are same positions in the unit_pos: \n {pd_ind[all_same.nonzero(as_tuple=True)[0]]}"
            )

        pd_off = unit_pos * box_size - pd_ind
        if vec is not None:
            feature = torch.cat([
                torch.ones(unit_pos.shape[0],
                           1,
                           dtype=torch.float,
                           device=unit_pos.device), pd_off, vec
            ],
                                dim=-1)
        else:
            feature = torch.cat([
                torch.ones(unit_pos.shape[0],
                           1,
                           dtype=torch.float,
                           device=unit_pos.device), pd_off
            ],
                                dim=-1)
        box[pd_ind[:, 0], pd_ind[:, 1], pd_ind[:, 2]] = feature
    else:
        box = np.zeros((*box_size, 4 + vec_dim))
        pd_ind = np.floor(np.clip(unit_pos, 0, 1 - 1E-7) *
                          box_size).astype(int)
        pd_off = unit_pos * box_size - pd_ind
        if vec is None:
            feature = np.concatenate([np.ones((unit_pos.shape[0], 1)), pd_off],
                                     axis=-1)
        else:
            feature = np.concatenate(
                [np.ones((unit_pos.shape[0], 1)), pd_off, vec], axis=-1)
        box[pd_ind[:, 0], pd_ind[:, 1], pd_ind[:, 2]] = feature
    return box


@torch.no_grad()
def box2atom(box,
             cell=[25.0, 25.0, 16.0],
             threshold=0.5,
             cutoff: float | tuple[float, ...] = 2.0,
             mode='O',
             nms=True,
             num_workers=0) -> list[Atoms] | Atoms:
    
    if box.dim() > 4:
        if num_workers == 0:
            return list(
                box2atom(b, cell, threshold, cutoff, mode=mode, nms=nms)
                for b in box)
        else:
            with Pool(num_workers) as p:
                return p.starmap(box2atom,
                                 [(b, cell, threshold, cutoff, mode, nms)
                                  for b in box])
    else:
        if mode == 'O':
            confidence, positions = box2orgvec(box,
                                               threshold,
                                               cutoff,
                                               cell,
                                               sort=True,
                                               nms=nms)
            if isinstance(positions, Tensor):
                confidence = confidence.detach().cpu().numpy()
                positions = positions.detach().cpu().numpy()
            atoms = Atoms("O" * positions.shape[0],
                          positions,
                          cell=cell,
                          pbc=False)
            atoms.set_array('conf', confidence)
        elif mode == 'OH':
            o_conf, o_pos = box2orgvec(box[..., :4],
                                       threshold,
                                       cutoff[0],
                                       cell,
                                       sort=True,
                                       nms=nms)
            h_conf, h_pos = box2orgvec(box[..., 4:],
                                       threshold,
                                       cutoff[1],
                                       cell,
                                       sort=True,
                                       nms=nms)
            atom1 = Atoms("O" * o_pos.shape[0], o_pos, cell=cell, pbc=False)
            atom1.set_array('conf', o_conf.numpy())
            atom2 = Atoms("H" * h_pos.shape[0], h_pos, cell=cell, pbc=False)
            atom2.set_array('conf', h_conf.numpy())
            atoms = atom1 + atom2
    return atoms


def makewater(pos: ndarray, rot: ndarray):
    # N 3, N 3 3 -> N 3 3
    if not isinstance(pos, ndarray):
        pos = pos.detach().cpu().numpy()
    if not isinstance(rot, ndarray):
        rot = rot.detach().cpu().numpy()

    water = np.array([
        [0., 0., 0.],
        [0., 0., 0.9572],
        [0.9266272, 0., -0.23998721],
    ])

    # print( np.einsum("ij,Njk->Nik", water, rot) )
    return np.einsum("ij,Njk->Nik", water, rot) + pos[:, None, :]


@torch.jit.script
def __encode_th(positions):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[..., 0:3], positions[..., 3:6], positions[..., 6:]
    u, v = u + v - 2 * o, u - v
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = v / torch.norm(v, dim=-1, keepdim=True)
    v = torch.where(v[..., 1].unsqueeze(-1) >= 0, v, -v)
    v = torch.where(v[..., 0].unsqueeze(-1) >= 0, v, -v)
    return torch.cat([o, u, v], dim=-1)


#@nb.njit(fastmath=True, cache=True)
def __encode_np(positions: ndarray):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[..., 0:3], positions[..., 3:6], positions[..., 6:]
    u, v = u + v - 2 * o, u - v
    u = u / np.expand_dims(((u**2).sum(axis=-1)**0.5), -1)
    v = v / np.expand_dims(((v**2).sum(axis=-1)**0.5), -1)
    v = np.where(v[..., 1][..., None] >= 0, v, -v)
    v = np.where(v[..., 0][..., None] >= 0, v, -v)
    return np.concatenate((o, u, v), axis=-1)


@torch.jit.script
def __decode_th(emb):
    o, u, v = emb[..., 0:3], emb[..., 3:6], emb[..., 6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return torch.cat([o, h1, h2], dim=-1)


# @nb.njit(fastmath=True,parallel=True, cache=True)
def __decode_np(emb):
    o, u, v = emb[..., 0:3], emb[..., 3:6], emb[..., 6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return np.concatenate((o, h1, h2), axis=-1)


def encodewater(positions):
    if isinstance(positions, Tensor):
        return __encode_th(positions)
    else:
        return __encode_np(positions)


def decodewater(emb):
    if isinstance(emb, Tensor):
        return __decode_th(emb)
    else:
        return __decode_np(emb)


def rotate(points, rotation_vector: ndarray):
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


def replicate(points: ndarray, times: list[int], offset: ndarray) -> ndarray:
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


def grid_to_water_molecule(grids,
                           cell=[25.0, 25.0, 16.0],
                           threshold=0.5,
                           cutoff=2.0,
                           flip_axis=[False, False, True]) -> Atoms:
    """
    Convert grids to atoms formats.

    Args:
        grids (Tensor | ndarray): shape: (X, Y, Z, C)

    Returns:
        atoms (ase.Atoms)
    """
    conf, pos, rotation = box2orgvec(grids,
                                     threshold=threshold,
                                     cutoff=cutoff,
                                     real_size=cell,
                                     sort=True,
                                     nms=True)
    rotation = rotation.view(-1, 9)[:, :6]

    atoms_pos = decodewater(np.concatenate([pos, rotation],
                                           axis=-1)).reshape(-1, 3)
    atoms_pos = atoms_pos * np.where(flip_axis, -1, 1) + np.where(
        flip_axis, cell, 0)

    atoms_types = ["O", "H", "H"] * len(pos)
    atoms_conf = np.repeat(conf, 3)
    atoms = Atoms(atoms_types,
                  atoms_pos,
                  tags=atoms_conf,
                  cell=cell,
                  pbc=False)

    return atoms


def combine_atoms(atom_list: list[Atoms], eps=0.5):
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


class view(object):

    @staticmethod
    def space_to_image(tensor: Tensor):
        "C H W D -> (C D) 1 H W"
        tensor = tensor.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)
        image = make_grid(tensor, nrow=int(np.sqrt(tensor.shape[0])))
        return image

    @staticmethod
    def afm_points_to_grid(afm: Tensor, grid: Tensor):
        # afm: X Y Z 1, grid: X Y Z 8
        grid = grid[..., (0, 4)].permute(2, 3, 1,
                                         0).max(0,
                                                keepdim=True).values  # 1 2 Y X
        afm = afm.permute(2, 3, 1, 0)  # Z 1 Y X
        grid = F.interpolate(grid,
                             afm.shape[2:]).repeat([afm.shape[0], 1, 1, 1])
        images = torch.cat([grid, afm], dim=1)  # Z 3 Y X
        images = make_grid(images)
        return images


class PointCloudProjector():

    def __init__(self,
                 real_size=(25.0, 25.0, 16.0),
                 box_size=(25, 25, 16),
                 kernel_size=5,
                 sigma=0.6):
        self.sigma = sigma
        self.dim = len(real_size)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * self.dim
        self.real_size = np.array(real_size)
        self.box_size = np.array(box_size)
        self.kernel_size = np.array(kernel_size)
        # (k * k * k) * 3
        self.k_ind_offset = np.stack(np.meshgrid(
            *[np.arange(-(k - 1) // 2, (k - 1) // 2 + 1) for k in kernel_size],
            indexing='ij'),
                                     axis=-1).reshape(-1, self.dim)
        # print(self.k_ind_offset.shape)
        self.k_pos_offset = self.k_ind_offset * (self.real_size /
                                                 self.box_size)
        self.k_pad = np.concatenate(
            [self.kernel_size // 2, (self.kernel_size - 1) // 2])
        # self.pad = [min(self.k_ind_offset),]
        # print(self.k_ind_offset.shape)

    def gaussian_function(self, x, mu, sigma, norm=True):
        # x: (*, 3)
        if norm:
            return np.prod(np.exp(-(x - mu)**2 / (2 * sigma**2)), axis=-1)
        else:
            return np.prod(1 / (sigma * np.sqrt(2 * np.pi)) *
                           np.exp(-(x - mu)**2 / (2 * sigma**2)),
                           axis=-1)

    def __call__(self, pc: np.ndarray):
        # pc: (*, N, 3)
        box_shape = [*pc.shape[:-2], *self.box_size]
        box = np.zeros(box_shape)
        pos_ind = (pc / self.real_size * self.box_size).astype(int)
        pos_offset = pc / self.real_size * self.box_size - pos_ind
        pos_ind = pos_ind[..., None, :] + self.k_ind_offset
        pos_offset = pos_offset[..., None, :] + self.k_pos_offset
        mask = (pos_ind >= 0).all(axis=-1) & (pos_ind
                                              < self.box_size).all(axis=-1)
        pos_ind = pos_ind[mask]
        batch_ind = np.nonzero(mask)[:-2]
        if len(batch_ind) > 0:
            batch_ind = np.stack(batch_ind, axis=-1)
            pos_ind = np.concatenate([batch_ind, pos_ind], axis=-1)

        pos_offset = pos_offset[mask] - 0.5 * self.box_size / self.real_size
        gauss_pos = self.gaussian_function(pos_offset, 0, self.sigma)

        np.add.at(box, tuple(pos_ind.T), gauss_pos)

        return box

def write_array(file: File, name: str, array: np.ndarray, property: dict = {}):
    dts = file.create_dataset(name, data=array)
    for key, val in property.items():
        dts.attrs[key] = val
    return dts

def write_dict(file: File, name: str, dic: dict, property: dict = {}):
    group = file.create_group(name)
    for key, val in dic.items():
        group[key] = val
    for key, val in property.items():
        group.attrs[key] = val
    return group

def load_by_name(file: File, name: str):
    atom_keys = ['positions', 'numbers', 'cell', 'pbc']
    dic = dict(file[name].items())
    dic.update(file[name].attrs.items())
    atom_dic = {key: val[...] for key, val in dic.items() if key in atom_keys}
    info_dic = {key: val if isinstance(val, str) else val[...] for key, val in dic.items() if key not in atom_keys}
    info_dic['name'] = name
    atom_dic['info'] = info_dic
    atoms = Atoms.fromdict(atom_dic)
    return atoms

def write_atoms(file: File, name: str, atoms: Atoms):
    atom_dic = atoms.todict()
    for val in atom_dic.values():
        if isinstance(val, np.ndarray):
            if val.dtype == np.float64:
                val = val.astype(np.float32)
            elif val.dtype == np.int64:
                val = val.astype(np.int32)

    return write_dict(file, name, atom_dic)

def list_names(file: File):
    return list(file.keys())

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
