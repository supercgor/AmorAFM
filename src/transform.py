import numpy as np
import random
import torch
import torchvision
from typing import TypeVar, Callable

def z_sampler(use: int, total: int, is_rand: bool = False) -> np.ndarray:
    """
    Sample the z-axis. if `is_rand`, the given index will be randomly sampled from the total index, else the index will be the first `use` index.

    Args:
        use (int): images to use, default is 10
        total (int): total images in the dataset, default is 20.
        is_rand (bool, optional): Whether to sample randomly from different height. Defaults to False.

    Returns:
        tuple[int]: the index of the z-axis.
    """
    if is_rand:
        sp = random.sample(range(total), k=(use % total))
        return np.asarray([i for i in range(total) for _ in range(use // total + (i in sp))])
    else:
        return np.asarray([i for i in range(total) for _ in range((use // total + ((use % total) > i)))])

def z_layerwise_sampler(use: list[int] = [4, 4, 3], 
                        split: list[int] = [10, 18], 
                        total: int = 25, 
                        is_rand: bool = False
                        ) -> np.ndarray:
    idxs = []
    split = [0] + split + [total]
    for i in range(len(use)):
        idxs.append(z_sampler(use[i], split[i+1] - split[i], is_rand) + split[i])
    return np.concatenate(idxs)

OptArray = TypeVar('NDArray')
OptAtoms = TypeVar('Atoms')
type transform_fn = Callable[[OptArray, OptAtoms], tuple[OptArray, OptAtoms]]

def noisy_fn(max_intensity: float | bool = 0.03) -> transform_fn:
    if isinstance(max_intensity, bool):
        max_intensity = 0.03
        
    def _fn(imgs, atoms):
        intensity = random.uniform(0, max_intensity)
        if random.randbytes(1):
            return imgs + np.random.randn(*imgs.shape) * intensity, atoms
        else:
            return imgs + np.random.uniform(-intensity, intensity, imgs.shape), atoms
    return _fn

def flip_fn(ratio: tuple[float, float] = (0.5, 0.5)) -> transform_fn:
    def _fn(imgs, atoms): # C Z Y X
        if random.random() > ratio[0]:
            imgs = imgs[:, :, ::-1]
            atoms.positions[:, 1] = atoms.cell.array[1, 1] - atoms.positions[:, 1]
            
        if random.random() > ratio[1]:
            imgs = imgs[:, ::-1]
            atoms.positions[:, 0] = atoms.cell.array[0, 0] - atoms.positions[:, 0]
        
        return imgs, atoms
    return _fn

def jitter_fn(b=0.1, c =0.1, s=0.1, h=0.1) -> transform_fn:
    t = torchvision.transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
    def _fn(imgs, atoms):
        imgs = torch.from_numpy(imgs).transpose(0, 1)
        imgs = t(imgs).clamp_(0, 1).transpose(0, 1).numpy()
        return imgs, atoms
    return _fn

def blur_fn(ksize: int = 3, sigma: float = 0.1) -> transform_fn:
    t = torchvision.transforms.GaussianBlur(ksize, (sigma * 0.5, sigma * 1.5))
    def _fn(imgs, atoms):
        imgs = torch.from_numpy(imgs).transpose(0, 1)
        imgs = t(imgs).transpose(0, 1).numpy()
        return imgs, atoms
    return _fn

def pixel_shift_fn(max_shift: tuple[float, float] = (0.1, 0.1), ref: int = 3) -> transform_fn:
    def _fn(imgs, atoms):
        C, Z, H, W = imgs.shape
        y_shift = int(random.uniform(-max_shift[0] * H, max_shift[0] * H)) / (Z - 1)
        x_shift = int(random.uniform(-max_shift[1] * W, max_shift[1] * W)) / (Z - 1)
        y_shift = (np.arange(Z) - ref) * y_shift
        x_shift = (np.arange(Z) - ref) * x_shift
        for i in range(Z):
            if i != ref:
                imgs[:, i] = np.roll(imgs[:, i], (y_shift[i], x_shift[i]), (-2, -1))
        return imgs, atoms
    return _fn

def normalize_fn() -> transform_fn:
    def _fn(imgs, atoms):
        MAX = np.max(imgs, axis = (0, -2, -1), keepdims = True)
        MIN = np.min(imgs, axis = (0, -2, -1), keepdims = True)
        imgs = (imgs - MIN) / (MAX - MIN)
        return imgs, atoms
    return _fn

def noise_label_fn(offset: tuple[float, float, float] = (0.0, 0.0, 0.0), sigma: tuple[float, float, float] = (0.0, 0.0, 0.1)):
    def _fn(imgs, atoms):
        noise = np.random.normal(offset, sigma, size=atoms.positions.shape)
        atoms.positions += noise
        reduce_pos = atoms.positions @ np.linalg.inv(atoms.cell.array)
        atoms = atoms[np.all(reduce_pos >= 0, axis = 1) & np.all(reduce_pos < 1, axis = 1)]
        return imgs, atoms
    return _fn

def cutout_fn(max_num: int = 4, min_scale: float= 0.05, max_area: float = 0.02, area_decay: float = 0.9) -> transform_fn:
    min_area = min_scale ** 2
    def _fn(imgs, atoms):
        C, Z, H, W = imgs.shape
        now_area = max_area
        for z in range(Z):
            if now_area < min_area:
                break
            
            for _ in range(random.randint(0, max_num)):
                area = ((random.random() ** 2) * now_area * (1 - min_area) + min_area) # the power is to make the area more likely to be small
                pic_x = ((random.random() ** 2) * (np.sqrt(area) - min_scale) + min_scale)
                pic_y = area / pic_x
                
                if random.getrandbits(1):
                    pic_x, pic_y = pic_y, pic_x

                pic_x = int(np.rint(pic_x * W))
                pic_y = int(np.rint(pic_y * H))
                
                start_x = random.randint(0, W - pic_x)
                start_y = random.randint(0, H - pic_y)
                
                m = imgs[:, z, start_y:start_y + pic_y, start_x:start_x + pic_x].mean()
                imgs[:, z, start_y:start_y + pic_y, start_x:start_x + pic_x] = m
                
            now_area *= area_decay
        return imgs, atoms
    return _fn

def random_remove_atoms_fn(cutoff: float = 12.0, ratio = 0.3):
    def _fn(imgs, atoms):
        atoms_top = atoms[atoms.positions[:, 2] > cutoff]
        atoms_bot = atoms[atoms.positions[:, 2] <= cutoff]
        remove_num = int(np.random.uniform(0, int(len(atoms_top) * ratio)))
        keep_ind = np.random.permutation(len(atoms_top))[:len(atoms_top) - remove_num]
        atoms_top = atoms_top[keep_ind]
        return imgs, atoms_top + atoms_bot
    return _fn