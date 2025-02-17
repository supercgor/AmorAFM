import h5py
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import torchvision

from ase import io, Atoms
from omegaconf import ListConfig
from PIL import Image
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms.functional import to_tensor
from typing import Literal, Iterable
from utils.aseio import load_by_name
from utils.lib import vec2box, view, PointCloudProjector

def z_sampler(use: int, total: int, is_rand: bool = False) -> list[int]:
    if is_rand:
        sp = random.sample(range(total), k=(use % total))
        return [i for i in range(total) for _ in range(use // total + (i in sp))]
    else:
        return [i for i in range(total) for _ in range((use // total + ((use % total) > i)))]

def layerz_sampler(use, total, is_rand, layer = [0, 5, 12]):
    out = []
    while layer[-1] > total:
        layer.pop()
    num_layer = len(layer)
    layer = [*layer, total]
    for i, (low, high) in enumerate(zip(layer[:-1], layer[1:])):
        sam = z_sampler((use // num_layer + ((use % num_layer) > i)), high - low, is_rand)
        out.extend([j + low for j in sam])
    return out

def BidxDataLoader(dataset, batch_size = 1, iters = None, shuffle = False, drop_last = False, num_workers = 0, pin_memory = False, *args, **kwargs):
    """
    This is an extension of torch.utils.data.DataLoader that allows for batched-indexing of the data and repeats iteration if iters is specified.
    This will significantly speed up data loading when batch_size is big and iters is much bigger than `len(dataset)//batch_size`.
    

    Args:
        dataset (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 1.
        iters (_type_, optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.
        drop_last (bool, optional): _description_. Defaults to False.
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    sampler = _BidxSampler(dataset, batch_size=batch_size, drop_last= drop_last, shuffle=shuffle)
    
    if iters is None:
        return DataLoader(dataset, num_workers=num_workers, sampler=sampler, collate_fn=_non_batched_fn, pin_memory=pin_memory, *args, **kwargs)
    else:
        return DataLoader(dataset, num_workers=num_workers, sampler=ItLoader(sampler, iters), collate_fn=_non_batched_fn, pin_memory=pin_memory, *args, **kwargs)

def _non_batched_fn(x):
    return x[0]

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

class _BidxSampler(Sampler):
    def __init__(self, dts, batch_size: int, drop_last: bool = False, shuffle: bool = False):
        self.dts = dts
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(len(self.dts))
        else:
            idxs = torch.arange(len(self.dts))
        idxs = torch.split(idxs, self.batch_size)
        if self.drop_last and len(idxs[-1]) < self.batch_size:
            idxs = idxs[:-1]
        return iter(idxs)



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

class Resize(nn.Module):
    def __init__(self, size: tuple[int] = (100, 100)):
        super().__init__()
        self.size = size
        
    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, self.size, mode='bilinear', align_corners=False)

def cutout(input: Tensor, 
           max_num: int = 3, 
           max_scale: tuple[float, float] = (0.1, 0.9), 
           size: float = 0.01, 
           layerwise: bool = True
           ) -> Tensor:
    """
    Cutout augmentation. Randomly cut out a rectangle from the image. The operation is in-place.

    Args:
        x (Tensor): _description_
        N (int, optional): _description_. Defaults to 4.
        scale (tuple[float, float], optional): _description_. Defaults to (0.02, 0.98).
        size (float, optional): _description_. Defaults to 0.02.
        layerwise (bool, optional): _description_. Defaults to True.

    Returns:
        Tensor: _description_
    """
    _, D, H, W = input.shape
    HW = torch.as_tensor([H, W], device=input.device, dtype=input.dtype)
    if not layerwise:
        D = 1
    input = input.permute(1, 2, 3, 0) # D, H, W, C
    try:
        usearea = torch.randn(D, max_num, device = input.device, dtype= input.dtype).abs() * size
        pos = torch.rand(D, max_num, 2, device = input.device, dtype= input.dtype)
        sizex = torch.randn(D, max_num, device = input.device, dtype= input.dtype).abs() * usearea.sqrt()
        sizey = usearea / sizex
        size_hw = torch.stack([sizex, sizey], dim=-1).clip(*max_scale)
        start_points = ((pos - size_hw / 2).clamp(0, 1) * HW).long()
        end_points = ((pos + size_hw / 2).clamp(0, 1) * HW).long()
        
        if layerwise:
            for i in range(D):
                xi, stp, enp = input[i], start_points[i], end_points[i]
                xmean = xi.mean().nan_to_num(0)
                for j in range(max_num):
                    xi[stp[j,0]:enp[j,0], stp[j,1]:enp[j,1]] = xmean
        else:
            xi, stp, enp = input, start_points[0], end_points[0]
            xmean = xi.mean().nan_to_num(0)
            for j in range(max_num):
                xi[:, stp[j,0]:enp[j,0], stp[j,1]:enp[j,1]] = xmean
    except Exception as e:
        print(e)
    input = input.permute(3, 0, 1, 2)
    return input

def noisy(input: Tensor, 
          intensity: float = 0.1, 
          noisy_mode: tuple[int, ...] = (0, 1, 2), 
          noisy_type: tuple[int, ...] = (0, 1), 
          layerwise: bool = True
          ) -> Tensor:
    """
    Add noise to the image. The operation is in-place.
    
    Args:
        x (Tensor): input tensor, shape (C, D, H, W)
        intensity (float): noise intensity. Defaults to 0.1.
        noisy_mode (tuple[int]): noise mode, 0: None, 1: add, 2: times. Defaults to (0, 1, 2).
        noisy_type (tuple[int]): noise type, 0: normal, 1: uniform. Defaults to (0, 1).
        layerwise (bool): whether to apply the noise according to the layers. Defaults to True.
    Returns:
        Tensor: output tensor, shape (C, D, H, W)
    """
    C, D, H, W = input.shape
    if not layerwise:
        D = 1
    input = input.permute(1, 2, 3, 0) # D, H, W, C
    n = torch.empty(H, W, C, device=input.device, dtype=input.dtype)
    noisy_modes = torch.randint(0, len(noisy_mode), (D,), device=input.device)
    noisy_types = torch.randint(0, len(noisy_type), (D,), device=input.device)
    if layerwise:
        for i in range(D):
            noisy, mode = noisy_type[noisy_types[i]], noisy_mode[noisy_modes[i]]
            if mode == 0:
                continue
            
            if noisy == 0:
                n.normal_(0.0, intensity)
            elif noisy == 1:
                n.uniform_(-intensity, intensity)
            
            if mode == 1:
                input[i] += n
            elif mode == 2:
                input[i] *= 1 + n
    else:
        noisy, mode = noisy_type[noisy_types[0]], noisy_mode[noisy_modes[0]]
        if mode != 0:
            if noisy == 0:
                n.normal_(0.0, intensity)
            elif noisy == 1:
                n.uniform_(-intensity, intensity)
                
            if mode == 1:
                input += n
            elif mode == 2:
                input *= 1 + n
                
    input.clamp_(0, 1)
    input = input.permute(3, 0, 1, 2)
    return input

class Cutout(nn.Module):
    def __init__(self, 
                 max_num: int = 4, 
                 max_scale: tuple[float, float] = (0.02, 0.98), 
                 size: float = 0.02, 
                 layerwise: bool = True
                 ):
        super().__init__()
        self.N = max_num
        self.scale = max_scale
        self.size = size
        self.layerwise = layerwise
        
    def forward(self, input: Tensor) -> Tensor:
        return cutout(input, self.N, self.scale, self.size, self.layerwise)

# class Noisy(nn.Module):
#     def __init__(self, 
#                  intensity: float = 0.1, 
#                  noisy_mode: list[int] = (0, 1, 2), 
#                  noisy_type: list[int] = (0, 1), 
#                  layerwise: bool = True
#                  ):
#         super().__init__()
#         self.intensity = intensity
#         self.noisy_mode = noisy_mode
#         self.noisy_type = noisy_type
#         self.layerwise = layerwise
        
#     def forward(self, x: Tensor) -> Tensor:
#         return noisy(x, self.intensity, self.noisy_mode, self.noisy_type, self.layerwise)

class Noisy(nn.Module):
    def __init__(self, intensity: float | bool = 0.1):
        super().__init__()
        if isinstance(intensity, bool):
            self.intensity = 0.1 if intensity else 0.0
        else:
            self.intensity = intensity
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Add noise to the afm images, the lower `r` layers are `+` mode, the upper `10-r` layers are `x` mode.

        Args:
            x (Tensor): Shape: [C, D, H, W]

        Returns:
            Tensor: _description_
        """
        
        noise = torch.randn_like(x)
        x *= 1 + noise * self.intensity
        # x = (x - x.amin((0,2,3), keepdim=True)) / (x.amax((0,2,3), keepdim=True) - x.amin((0,2,3), keepdim=True))
        return x
        
class Flip(nn.Module):
    def __init__(self,
                 ratio = 0.5,
                 axis = 1
                 ):
        self.ratio = ratio
        self.axis = axis
    
    def forward(self, inputs):
        if np.random.rand() > self.ratio:
            return torch.flip(inputs, [self.axis])
        else:
            return inputs
        

class ColorJitter(nn.Module):
    def __init__(self, 
                 brightness: float | tuple[float, float] = 0.1, 
                 contrast: float | tuple[float, float] = 0.1, 
                 saturation: float | tuple[float, float] = 0.1,
                 hue: float | tuple[float, float] = 0.1
                 ):
        super().__init__()
        self.jt = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2, 3)
        x = self.jt(x)
        x = x.permute(1, 0, 2, 3)
        return x.clamp_(0, 1)

class Blur(nn.Module):
    def __init__(self, 
                 ksize=5, 
                 sigma: float = 2.0
                 ):
        super().__init__()
        self.bur = torchvision.transforms.GaussianBlur(ksize, (sigma * 0.5, sigma * 1.5))
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2, 3)
        x = self.bur(x)
        x = x.permute(1, 0, 2, 3)
        return x

def pixel_shift(input: Tensor, 
                max_shift=(0.1, 0.1), 
                fill: Literal['none', 'mean'] | float = "none", 
                ref: int = 3, 
                layerwise: bool = False
                ) -> Tensor:
    C, D, H, W = input.shape
    input = input.permute(1, 0, 2, 3)
    max_shift = torch.as_tensor(max_shift, device=input.device, dtype=input.dtype)

    if layerwise:
        max_shift = torch.rand(D, 2, device=input.device, dtype=input.dtype) * max_shift * 2 - max_shift
        shift = (max_shift * torch.as_tensor([H, W], device=input.device, dtype=input.dtype)).long()
    else:
        max_shift = torch.rand(2, device=input.device, dtype=input.dtype) * max_shift * 2 - max_shift
        max_shift = max_shift / (D - 1)
        shift = torch.arange(D, device=input.device, dtype=input.dtype) - ref
        shift = shift[:, None] * max_shift[None, :] # D, 2
        shift = (shift * torch.as_tensor([H, W], device=input.device, dtype=input.dtype)).long()
    for i in range(D):
        if i == ref:
            continue
        input[i] = torch.roll(input[i], tuple(shift[i]), (-2, -1))
        if fill != "none":
            if fill == "mean":
                fill = input[i].mean()
            if shift[i, 0] > 0:
                input[i, :, :shift[i, 0], :] = fill
            elif shift[i, 0] < 0:
                input[i, :, shift[i, 0]:, :] = fill
            if shift[i, 1] > 0:
                input[i, :, :, :shift[i, 1]] = fill
            elif shift[i, 1] < 0:
                input[i, :, :, shift[i, 1]:] = fill
    input = input.permute(1, 0, 2, 3)
    return input

class MinMaxNormalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return (x - x.amin((0,2,3), keepdim=True)) / (x.amax((0,2,3), keepdim=True) - x.amin((0,2,3), keepdim=True))

class PixelShift(nn.Module):
    def __init__(self, 
                 max_shift=(0.1, 0.1), 
                 fill: Literal['none', 'mean'] | float = "none", 
                 ref: int = 3, 
                 layerwise: bool = False
                 ):
        super().__init__()
        self.max_shift = max_shift
        self.fill = fill
        self.ref = ref
        self.layerwise = layerwise

    def forward(self, x: Tensor) -> Tensor:  # shape (B ,C, X, Y)
        return pixel_shift(x, self.max_shift, self.fill, self.ref, self.layerwise)


class LabelZNoise(nn.Module):
    def __init__(self, sigma = 0.1):
        super().__init__()
        self._sigma = sigma
        
    def forward(self, x):
        # x: N * 9
        N, _ = x.shape
        noise = torch.zeros_like(x)
        noise[:, 2].normal_(0, self._sigma)
        noise[:, (5,8)] = noise[:, (2,)]
        x = x + noise
        return x
    

class DetectDataset(Dataset):
    def __init__(self, 
                 fname: str,
                 mode: Literal['afm+label', 'afm', 'label', 'hdf5'] = 'afm+label',
                 num_images: int | list[int]                        = 10,
                 image_split: list[int] | None                      = None,
                 image_size: tuple[int, int]                        = (100, 100),
                 real_size: tuple[float, float, float]              = (25.0, 25.0, 3.0),
                 box_size: tuple[int, int, int]                     = (32, 32, 4),
                 elements: tuple[int, ...]                          = (8, 1),
                 flipz: bool                                        = False,
                 z_align: str                                       = 'bottom',
                 cutinfo_path: str                                  = "",
                 random_transform: bool                             = True,
                 random_noisy: float | bool                         = True,
                 random_cutout                                      = True,
                 random_jitter                                      = True,
                 random_blur                                        = 2.0,
                 random_shift                                       = True,
                 random_flipx                                       = False,
                 random_flipy                                       = False,
                 random_zoffset                                     = 0,
                 random_top_remove_ratio                            = 0,
                 ):
        self.fname = fname
        self.num_images = num_images
        self.image_size = image_size
        self.image_split = image_split
        self.real_size = real_size
        self.box_size = box_size
        self.cutinfo_path = cutinfo_path 
        self.random_transform = random_transform    
        self.random_noisy = random_noisy
        self.random_cutout = random_cutout
        self.random_jitter = random_jitter
        self.random_blur = random_blur
        self.random_shift = random_shift
        self.random_flipx = random_flipx
        self.random_flipy = random_flipy
        self.random_zoffset = random_zoffset
        self.random_top_remove_ratio = random_top_remove_ratio

        # keys: list of file names or list of tuples (afm, label)
        self.mode = mode
        self.elements = elements
        self.z_align = z_align
        self.flipz = flipz
        
        if self.cutinfo_path:
            self.cutinfo = pd.read_csv(self.cutinfo_path)
            names, attrs = self.cutinfo.iloc[:, 0], self.cutinfo.iloc[:, 1:]
            self.cutinfo = {name: {attr: value for attr, value in zip(attrs.columns, values)} for name, values in zip(names, attrs.values)}

        if mode == 'afm+label':
            self.keys = []
            for afm_dir in os.listdir(os.path.join(fname, 'afm')):
                if os.path.isdir(os.path.join(fname, 'afm', afm_dir)):
                    if os.path.exists(os.path.join(fname, 'label', f"{afm_dir}.poscar")):
                        self.keys.append((os.path.join(fname, 'afm', afm_dir), os.path.join(fname, 'label', f"{afm_dir}.poscar")))
                    elif os.path.exists(os.path.join(fname, 'label', f"{afm_dir}.xyz")):
                        self.keys.append((os.path.join(fname, 'afm', afm_dir), os.path.join(fname, 'label', f"{afm_dir}.xyz")))
                    else:
                        continue
                    
        elif mode == 'label':
            self.keys = []
            for label in os.listdir(fname):
                if label.endswith('.xyz') or label.endswith('.poscar'):
                    self.keys.append(os.path.join(fname, label))
        
        elif mode == 'afm':
            self.keys = []
            for afm_dir in os.listdir(fname):
                if os.path.isdir(os.path.join(fname, afm_dir)):
                    self.keys.append(os.path.join(fname, afm_dir))
            
        elif mode == 'hdf5':
            with h5py.File(fname, 'r') as f:
                self.keys = list(f.keys())
        
        # initialize transform
        if not self.random_transform:
            # self.transform = nn.Sequential()
            self.transform = MinMaxNormalize()
        else:
            self.transform = nn.Sequential()
            
            if self.random_shift:
                self.transform.append(PixelShift())
            if self.random_cutout:
                self.transform.append(Cutout())
            if self.random_jitter:
                self.transform.append(ColorJitter())
            if self.random_noisy:
                self.transform.append(Noisy(intensity = self.random_noisy if isinstance(self.random_noisy, float) else 0.03))
            if self.random_blur:
                self.transform.append(Blur(sigma = self.random_blur if isinstance(self.random_blur, float) else 1.0))
                
            self.transform.append(MinMaxNormalize())
    
    def key_filter(self, func):
        self.keys = list(filter(func, self.keys))
    
    def _read_hdf(self, idx):
        with h5py.File(self.fname, 'r') as f:
            fname = self.keys[idx]
            atoms = load_by_name(f, fname)
        # if len(atoms) == 0:
        #     atoms = None
    
        if 'img' in atoms.info:
            afm = atoms.info.pop('img')
            if isinstance(self.num_images, int):
                image_idx = z_sampler(self.num_images, len(afm), is_rand = self.random_transform)
            elif isinstance(self.num_images, Iterable):
                if (cutname:= fname.split('_')[0]) in self.cutinfo:
                    split = [self.cutinfo[cutname]['low'], self.cutinfo[cutname]['up']]
                elif self.image_split is not None:
                    split = self.image_split
                else:
                    raise ValueError('image_split should be provided')
                image_idx = z_layerwise_sampler(self.num_images, split, len(afm), is_rand = self.random_transform)
            else:
                raise ValueError('num_images should be int or list of int with image_split')
            afm = afm[image_idx]
            afm = torch.as_tensor(afm / 255.0, dtype=torch.float)
            afm = afm.permute(1, 0, 2, 3) #
        else:
            afm = None
        return fname, atoms, afm

    def _read_afm(self, path):
        images_name = list(os.listdir(path))
        images_name = list(filter(lambda name: name.endswith(('.png', '.jpg')), images_name))
        images_name.sort(key = lambda name: int(name.split('.')[0]))
                    
        afm = []
        for image_name in images_name:
            afm_image = Image.open(os.path.join(path, image_name)).convert('L')
            afm_image = afm_image.transpose(method=Image.Transpose.TRANSPOSE)
            afm_image = afm_image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            afm_image = afm_image.resize(self.image_size)
            afm_image = to_tensor(afm_image) # C X Y
            afm.append(afm_image)
        
        afm = torch.stack(afm, dim = 0) # N C X Y
        if isinstance(self.num_images, int):
            image_idx = z_sampler(self.num_images, len(afm), is_rand = self.random_transform)
        elif isinstance(self.num_images, (ListConfig, list)):
            if self.cutinfo_path and (cutname:= os.path.basename(path).split('_')[0]) in self.cutinfo:
                split = [self.cutinfo[cutname]['low'], self.cutinfo[cutname]['up']]
            elif self.image_split is not None:
                split = self.image_split
            else:
                raise ValueError('image_split should be provided')
            image_idx = z_layerwise_sampler(self.num_images, split, len(afm), is_rand = self.random_transform)
        else:
            raise ValueError('num_images should be int or list of int with image_split')
        afm = afm[image_idx]

        afm = afm.permute(1, 0, 2, 3) # N C X Y -> C N X Y
        return afm
        
    def _read_label(self, path) -> Atoms:
        atoms: Atoms = io.read(path)
        if self.real_size is not None:
            atoms.set_cell(np.diag(self.real_size))

        return atoms
   
    def __getitem__(self, idx):
        # fname: str
        # afm: Tensor X Y Z C
        # grids: Tensor X Y Z C
        # atoms: Atoms
        if self.mode == 'hdf5':
            fname, atoms, afm = self._read_hdf(idx)
        elif self.mode == 'afm+label':
            afm_dir, label = self.keys[idx]
            fname = os.path.splitext(os.path.basename(afm_dir))[0]
            afm = self._read_afm(afm_dir)
            atoms = self._read_label(label)
        elif self.mode == 'afm':
            afm_dir = self.keys[idx]
            fname = os.path.splitext(os.path.basename(afm_dir))[0]
            afm = self._read_afm(afm_dir)
            atoms = None
        elif self.mode == 'label':
            label = self.keys[idx]
            fname = os.path.splitext(os.path.basename(label))[0]
            afm = None
            atoms = self._read_label(label)
                
        flip = [False, False, self.flipz]
        
        # random transform
        if afm is not None:
            if self.random_transform:
                afm = self.transform(afm)
                if self.random_flipx and np.random.rand() > 0.5:
                    afm = torch.flip(afm, [2])
                    flip[0] = True
                if self.random_flipy and np.random.rand() > 0.5:
                    afm = torch.flip(afm, [3])
                    flip[1] = True
        
            afm = afm.permute(2, 3, 1, 0) # C N X Y -> X Y Z C
        # load labels
        
        if atoms is None:
            return fname, afm, None, None
        
        else:
            for i, flip_axis in enumerate(flip):
                if flip_axis:
                    atoms.positions[:,i] = atoms.cell[i,i] - atoms.positions[:,i]
            
            if self.random_transform:
                if self.random_zoffset != 0: # [-1.5, -.5]
                    z_offset = np.random.uniform(*self.random_zoffset)
                    atoms.positions[:, 2] += z_offset
                        
            if self.z_align == 'top':
                atoms.positions[:, 2] += self.box_size[2] - self.real_size[2]
            
            elif self.z_align == 'bottom':
                pass
            
            if self.random_transform:
                if self.random_top_remove_ratio > 0:
                    atoms_top = atoms[atoms.positions[:, 2] > 12.0]
                    atoms_bot = atoms[atoms.positions[:, 2] <= 12.0]
                    remove_num = int(np.random.uniform(0, int(len(atoms_top) * self.random_top_remove_ratio)))
                    keep_ind = np.random.permutation(len(atoms_top))[:len(atoms_top) - remove_num]
                    atoms_top = atoms_top[keep_ind]
                    atoms = Atoms(cell = atoms.cell, 
                                positions = np.concatenate([atoms_top.positions, atoms_bot.positions], axis = 0), 
                                numbers = np.concatenate([atoms_top.numbers, atoms_bot.numbers], axis = 0),
                                info = atoms.info
                                )
                if self.random_zoffset != 0:
                    atoms = atoms[(atoms.positions[:, 2] > 0) & (atoms.positions[:, 2] < self.real_size[2])]
                
            atom_pos = atoms.positions / np.diagonal(atoms.cell)

            box_pos = []
            for element in self.elements:
                elem_pos = atom_pos[atoms.numbers == element]
                elem_box = vec2box(elem_pos, None, self.box_size)
                box_pos.append(elem_box)
            grids = np.concatenate(box_pos, axis = -1)
            grids = torch.as_tensor(grids, dtype=torch.float)
            
            return fname, afm, grids, atoms

    def __len__(self):
        return len(self.keys)

def collate_fn(batch):
    fname, afm, grids, atoms = zip(*batch)
    afm = afm if afm[0] is None else torch.stack(afm)
    grids = grids if grids[0] is None else torch.stack(grids)
    return fname, afm, grids, atoms

class DetectDataset(Dataset):
    def __init__(self, 
                 fname: str,
                 mode: Literal['afm+label', 'afm', 'label', 'hdf5'] = 'afm+label',
                 num_images: int = 10,
                 image_size: tuple[int, int] = (100, 100),
                 real_size: tuple[float, float, float] = (25.0, 25.0, 3.0),
                 box_size: tuple[int, int, int] = (32, 32, 4),
                 elements: tuple[int, ...] = (8, 1),
                 flipz: bool = False,
                 z_align: str = 'bottom',
                 random_transform: bool = True,
                 random_noisy = True,
                 random_cutout = True,
                 random_jitter = True,
                 random_blur = True,
                 random_shift = True,
                 random_flipx = False,
                 random_flipy = False,
                 random_zoffset = None,
                 random_top_remove_ratio = None,
                 ):
        self.fname = fname
        self.num_images = num_images
        self.image_size = image_size
        self.real_size = real_size
        self.box_size = box_size
        self.random_transform = random_transform
        self.random_noisy = random_noisy
        self.random_cutout = random_cutout
        self.random_jitter = random_jitter
        self.random_blur = random_blur
        self.random_shift = random_shift
        self.random_flipx = random_flipx
        self.random_flipy = random_flipy
        self.random_zoffset = random_zoffset
        self.random_top_remove_ratio = random_top_remove_ratio
        
        self.pcp = PointCloudProjector(real_size, box_size, sigma = 0.6)
        
        # keys: list of file names or list of tuples (afm, label)
        self.mode = mode
        self.elements = elements
        self.z_align = z_align
        self.flipz = flipz
        
        if mode == 'afm+label':
            self.keys = []
            for afm_dir in os.listdir(os.path.join(fname, 'afm')):
                if os.path.isdir(os.path.join(fname, 'afm', afm_dir)):
                    if os.path.exists(os.path.join(fname, 'label', f"{afm_dir}.poscar")):
                        self.keys.append((os.path.join(fname, 'afm', afm_dir), os.path.join(fname, 'label', f"{afm_dir}.poscar")))
                    elif os.path.exists(os.path.join(fname, 'label', f"{afm_dir}.xyz")):
                        self.keys.append((os.path.join(fname, 'afm', afm_dir), os.path.join(fname, 'label', f"{afm_dir}.xyz")))
                    else:
                        continue
                    
        elif mode == 'label':
            self.keys = []
            for label in os.listdir(fname):
                if label.endswith('.xyz') or label.endswith('.poscar'):
                    self.keys.append(os.path.join(fname, label))
        
        elif mode == 'afm':
            self.keys = []
            for afm_dir in os.listdir(fname):
                if os.path.isdir(os.path.join(fname, afm_dir)):
                    self.keys.append(os.path.join(fname, afm_dir))
        
        elif mode == 'hdf5':
            with h5py.File(fname, 'r') as f:
                self.keys = list(f.keys())
        
        # initialize transform
        if not self.random_transform:
            self.transform = nn.Identity()
        else:
            self.transform = nn.Sequential()
            if self.random_shift is not None:
                self.transform.append(PixelShift())
            if self.random_cutout is not None:
                self.transform.append(Cutout())
            if self.random_jitter is not None:
                self.transform.append(ColorJitter())
            if self.random_noisy is not None:
                self.transform.append(Noisy())
            if self.random_blur is not None:
                self.transform.append(Blur())
    
    def key_filter(self, func):
        self.keys = list(filter(func, self.keys))
    
    def _read_hdf(self, idx):
        with h5py.File(self.fname, 'r') as f:
            fname = self.keys[idx]
            atoms = load_by_name(f, fname)
        if len(atoms) == 0:
            atoms = None
    
        if 'img' in atoms.info:
            afm = atoms.info.pop('img')
            image_idx = z_sampler(self.num_images, len(afm), is_rand = True)
            afm = afm[image_idx]
            afm = torch.as_tensor(afm / 255.0, dtype=torch.float)
            afm = afm.permute(1, 0, 2, 3) #
            
            
        else:
            afm = None
        return fname, atoms, afm

    def _read_afm(self, path):
        images_name = list(os.listdir(path))
        images_name = list(filter(lambda name: name.endswith(('.png', '.jpg')), images_name))
        images_name.sort(key = lambda name: int(name.split('.')[0]))
                    
        afm = []
        for image_name in images_name:
            afm_image = Image.open(os.path.join(path, image_name)).convert('L')
            afm_image = afm_image.transpose(method=Image.Transpose.TRANSPOSE)
            afm_image = afm_image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            afm_image = afm_image.resize(self.image_size)
            afm_image = to_tensor(afm_image) # C X Y
            afm.append(afm_image)
        
        afm = torch.stack(afm, dim = 0) # N C X Y
        image_idx = z_sampler(self.num_images, len(afm), is_rand = True)
        afm = afm[image_idx]

        afm = afm.permute(1, 0, 2, 3) # N C X Y -> C N X Y
        return afm
        
    def _read_label(self, path):
        atoms = io.read(path)
        if self.real_size is not None:
            atoms.set_cell(np.diag(self.real_size))

        return atoms
   
    def __getitem__(self, idx):
        # fname: str
        # afm: Tensor X Y Z C
        # grids: Tensor X Y Z C
        # atoms: Atoms
        if self.mode == 'hdf5':
            fname, atoms, afm = self._read_hdf(idx)
        elif self.mode == 'afm+label':
            afm_dir, label = self.keys[idx]
            fname = os.path.splitext(os.path.basename(afm_dir))
            afm = self._read_afm(afm_dir)
            atoms = self._read_label(label)
        elif self.mode == 'afm':
            afm_dir = self.keys[idx]
            fname = os.path.splitext(os.path.basename(afm_dir))
            afm = self._read_afm(afm_dir)
            atoms = None
        elif self.mode == 'label':
            label = self.keys[idx]
            fname = os.path.splitext(os.path.basename(label))
            afm = None
            atoms = self._read_label(label)
                
        flip = [False, False, self.flipz]
        
        # random transform
        if afm is not None:
            if self.random_transform:
                afm = self.transform(afm)
                if self.random_flipx and np.random.rand() > 0.5:
                    afm = torch.flip(afm, [2])
                    flip[0] = True
                if self.random_flipy and np.random.rand() > 0.5:
                    afm = torch.flip(afm, [3])
                    flip[1] = True
        
            afm = afm.permute(2, 3, 1, 0) # C N X Y -> X Y Z C
        # load labels
        
        if atoms is None:
            return fname, afm, None, None
        
        else:
            for i, flip_axis in enumerate(flip):
                if flip_axis:
                    atoms.positions[:,i] = atoms.cell[i,i] - atoms.positions[:,i]
            
            if self.random_transform:
                if self.random_zoffset is not None: # [-1.5, -.5]
                    z_offset = np.random.uniform(*self.random_zoffset)
                    atoms.positions[:, 2] += z_offset
                        
            if self.z_align == 'top':
                atoms.positions[:, 2] += self.box_size[2] - self.real_size[2]
            
            elif self.z_align == 'bottom':
                pass
            
            if self.random_top_remove_ratio is not None:
                atoms_top = atoms[atoms.positions[:, 2] > 12.0]
                atoms_bot = atoms[atoms.positions[:, 2] <= 12.0]
                remove_num = int(np.random.uniform(0, int(len(atoms_top) * self.random_top_remove_ratio)))
                keep_ind = np.random.permutation(len(atoms_top))[:len(atoms_top) - remove_num]
                atoms_top = atoms_top[keep_ind]
                atoms = Atoms(cell = atoms.cell, 
                              positions = np.concatenate([atoms_top.positions, atoms_bot.positions], axis = 0), 
                              numbers = np.concatenate([atoms_top.numbers, atoms_bot.numbers], axis = 0),
                              info = atoms.info
                              )
            
            atoms = atoms[(atoms.positions[:, 2] > 0) & (atoms.positions[:, 2] < self.real_size[2])]
            
            atom_pos = atoms.positions / np.diagonal(atoms.cell)

            box_pos = []
            box_vox = []
            for element in self.elements:
                elem_pos = atom_pos[atoms.numbers == element]
                # print(self.box_size, self.real_size, elem_pos)
                box_vox.append(vec2box(elem_pos, None, box_size=self.box_size))
                box_pos.append(self.pcp(atoms[atoms.numbers == element].positions))
            gau_grid = np.stack(box_pos, axis = -1)
            grids = np.concatenate(box_vox, axis = -1)
            
            gau_grid = torch.as_tensor(gau_grid, dtype=torch.float).clamp(0, 1)
            grids = torch.as_tensor(grids, dtype=torch.float)
            
            return fname, gau_grid, grids, atoms

    def __len__(self):
        return len(self.keys)

def collate_fn(batch):
    fname, afm, grids, atoms = zip(*batch)
    afm = afm if afm[0] is None else torch.stack(afm)
    grids = grids if grids[0] is None else torch.stack(grids)
    return fname, afm, grids, atoms