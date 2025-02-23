import numpy as np
import os
import pandas as pd
import torch

from ase import io, Atoms
from PIL import Image
from torch.utils.data import Dataset
from typing import Literal
from utils import vec2box
from pathlib import Path

from .transform import blur_fn, cutout_fn, jitter_fn, noisy_fn, noise_label_fn, normalize_fn, pixel_shift_fn, random_remove_atoms_fn, z_layerwise_sampler, z_sampler

class DetectDataset(Dataset):
    def __init__(self, 
                 fname: str,
                 mode: Literal['afm+label', 'afm', 'label']     = 'afm+label',
                 num_images: int | list[int]                    = 10,
                 image_split: None | list[int]                  = None,
                 image_size: tuple[int, int]                    = (100, 100),
                 real_size: tuple[float, ...]                   = (25.0, 25.0, 3.0),
                 box_size: tuple[int, ...]                      = (32, 32, 4),
                 elements: tuple[int, ...]                      = (8, 1),
                 flipz: bool                                    = False,
                 z_align: str                                   = 'bottom',
                 cutinfo_path: str                              = "",
                 random_transform: bool                         = True,
                 random_noisy: float | bool                     = True,
                 random_cutout                                  = True,
                 random_jitter                                  = True,
                 random_blur                                    = 2.0,
                 random_shift                                   = True,
                 random_flipx                                   = False,
                 random_flipy                                   = False,
                 random_zoffset                                 = None,
                 random_top_remove_ratio                        = 0.0,
                 ):
        self.fname = Path(fname)
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
            afm_dirs = set(self.fname.glob("afm/*"))
            label_files = self.fname.glob(r"label/*.*")
            
            self.keys = []
            for l in label_files:
                afm_dir = l.parents[1] / "afm" / l.with_suffix('').stem
                if afm_dir in afm_dirs:
                    self.keys.append((afm_dir, l))
                    
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
        
        self.transform = []
        
        if self.random_transform and "afm" in self.mode:
            if self.random_shift:
                self.transform.append(pixel_shift_fn())
            if self.random_cutout:
                self.transform.append(cutout_fn())
            if self.random_jitter:
                self.transform.append(jitter_fn())
            if self.random_noisy:
                self.transform.append(noisy_fn(self.random_noisy))
            if self.random_blur:
                self.transform.append(blur_fn())
                
        if self.random_transform and "label" in self.mode:
            if self.random_noisy:
                self.transform.append(noise_label_fn())
            if self.random_top_remove_ratio > 0:
                self.transform.append(random_remove_atoms_fn())
                
        if not self.transform:
            self.transform.append(normalize_fn())
    
    def sample(self, L):
        if isinstance(self.num_images, int):
            return z_sampler(self.num_images, L, is_rand = self.random_transform)
        elif isinstance(self.num_images, list) and isinstance(self.image_split, list):
            return z_layerwise_sampler(self.num_images, self.image_split, L, is_rand = self.random_transform)
        else:
            raise ValueError('num_images should be int or list of int with image_split')
        
    def key_filter(self, func):
        self.keys = list(filter(func, self.keys))
    
    def _read_afm(self, path):
        path = Path(path)
        files = sorted(filter(lambda x: x.suffix.endswith(('.png','.jpg')), path.iterdir()), key = lambda x: int(x.stem.split('.')[0]))
        idxs = self.sample(len(files))
        afms = []
        for i in idxs:
            img = Image.open(files[i]).convert('L').transpose(Image.Transpose.ROTATE_270)
            img = np.array(img.resize(self.image_size))
            afms.append(img)
            
        afm = np.stack(afms, axis = 0) / 255.0 # shape: (Z, Y, X), dtype: float
        return afm[None] # C Z Y X
        
    def _read_label(self, path) -> Atoms:
        atoms: Atoms = io.read(path) # type: ignore
        if self.real_size is not None:
            atoms.set_cell(np.diag(self.real_size))

        return atoms
   
    def __getitem__(self, idx):
        # fname: str
        # afm: Tensor X Y Z C
        # grids: Tensor X Y Z C
        # atoms: Atoms
        if self.mode == 'afm+label':
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
        
        for t in self.transform:
            afm, atoms = t(afm, atoms)
        
        if afm is not None:
            afm = torch.as_tensor(afm, dtype = torch.float32)
    
        if atoms is not None:
            box_pos = []
            inv_cell = np.linalg.inv(atoms.cell.array)
            for element in self.elements:
                elem_pos = atoms[atoms.numbers == element].positions @ inv_cell # type: ignore
                elem_box = vec2box(elem_pos, None, self.box_size)
                box_pos.append(elem_box)
            grids = np.concatenate(box_pos, axis = -1)
            grids = torch.as_tensor(grids, dtype = torch.float32)
        else:
            grids = None
        
        
        return fname, afm, grids, atoms

    def __len__(self):
        return len(self.keys)
    
    @staticmethod
    def collate_fn(batch):
        fname, afm, grids, atoms = zip(*batch)
        afm = afm if afm[0] is None else torch.stack(afm)
        grids = grids if grids[0] is None else torch.stack(grids)
        return fname, afm, grids, atoms