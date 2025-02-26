import numpy as np
import json
import os
import pandas as pd
import torch

from ase import io, Atoms
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import grid_sample
from typing import Literal
from pathlib import Path

from .utils import vec2box, make_grid_centers, make_grid_samples_points, masknms
from .transform import blur_fn, cutout_fn, jitter_fn, noisy_fn, noise_label_fn, normalize_fn, pixel_shift_fn, random_remove_atoms_fn, z_layerwise_sampler, z_sampler

class DetectDataset(Dataset):
    def __init__(self, 
                 fname: str | Path,
                 mode: Literal['afm+label', 'afm', 'label', 'afm+crop'] = 'afm+label',
                 num_images: int | list[int]                    = 10,
                 image_split: None | list[int]                  = None,
                 image_size: tuple[int, int]                    = (100, 100),       # (H, W)
                 real_size: tuple[float, float, float]          = (25.0, 25.0, 3.0),
                 box_size: tuple[int, int, int]                 = (32, 32, 4),
                 elements: tuple[int, ...]                      = (8, 1),
                 flipz: bool                                    = False,
                 z_align: str                                   = 'bottom',
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
                 normalize: bool                                = True,
                 ):
        self.fname = Path(fname)
        self.num_images = num_images
        self.image_size = image_size
        self.image_split = image_split
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
        self.normalize = normalize
        
        # keys: list of file names or list of tuples (afm, label)
        self.mode = mode
        self.elements = elements
        self.z_align = z_align
        self.flipz = flipz
        
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
        
        elif mode == 'afm+crop':
            cell_info = self.read_cell_info(self.fname)
            if cell_info is None:
                raise ValueError("No cell information found")
            
            self.cell_info = cell_info
            self.grids_centers = make_grid_centers(real_size[:2], cell = cell_info[1:], padding = 20.0, rot = 0.0, mode = 'inner')
            
            self.keys = [(self.fname, None)] * len(self.grids_centers)
        
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
                
        if "afm" in self.mode and self.normalize:
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
    
    def _read_afm_crop(self, idx):
        path = self.keys[idx][0]
        files = sorted(filter(lambda x: x.suffix.endswith(('.png','.jpg')) and x.stem.isdigit(), path.iterdir()), key = lambda x: int(x.stem.split('.')[0]))
        idxs = self.sample(len(files))
        afms = []
        grid_center = self.grids_centers[idx]
        grids = make_grid_samples_points(grid_center = grid_center[::-1], 
                                         bbox_size = self.real_size, 
                                         resolution = self.image_size, 
                                         cell = self.cell_info[1:], 
                                         rot = 270)
        grids = torch.as_tensor(grids)
        
        for i in idxs:
            img = Image.open(files[i]).convert('L')          
            img = np.array(img)
            afms.append(img)
            
        afm = np.stack(afms, axis = 0) / 255.0 # shape: (Z, Y, X), dtype: float
        
        afm = grid_sample(torch.as_tensor(afm)[None], grids[None], align_corners=False, padding_mode='border')[0].numpy()
        
        return afm[None] # C Z Y X
    
    def _read_label(self, path) -> Atoms:
        atoms: Atoms = io.read(path) # type: ignore
        if self.real_size is not None:
            atoms.set_cell(np.diag(self.real_size))

        return atoms
    
    def combine_label_crop(self, results: list[Atoms], nms = (2.0, 1.0), sort = True):
        assert len(results) == len(self.keys)
        atoms_confs = []
        atoms_types = []
        atoms_pos = []
        for atoms, grid_center in zip(results, self.grids_centers):
            pos = atoms.get_positions()
            offset_x = grid_center[0] - self.real_size[0] / 2
            offset_y = self.cell_info[2,1] - grid_center[1] - self.real_size[1] / 2
            pos[:, :2] += [offset_x, offset_y]
            
            atoms_types.append(atoms.numbers)
            atoms_confs.append(atoms.get_array('conf'))
            atoms_pos.append(pos)
        
        atoms_confs = np.concatenate(atoms_confs, axis = 0)
        atoms_types = np.concatenate(atoms_types, axis = 0)
        atoms_pos = np.concatenate(atoms_pos, axis = 0)
        
        if sort:
            idxs = np.argsort(atoms_confs)[::-1]
            atoms_confs = atoms_confs[idxs]
            atoms_types = atoms_types[idxs]
            atoms_pos = atoms_pos[idxs]
        
        if nms is not False:
            o_confs = atoms_confs[atoms_types == 8]
            o_pos = atoms_pos[atoms_types == 8]
            
            mask = masknms(o_pos, nms[0] if isinstance(nms, tuple) else nms)
            
            o_confs = o_confs[mask]
            o_pos = o_pos[mask]
            
            h_confs = atoms_confs[atoms_types == 1]
            h_pos = atoms_pos[atoms_types == 1]
            
            mask = masknms(h_pos, nms[1] if isinstance(nms, tuple) else nms)
            
            h_confs = h_confs[mask]
            h_pos = h_pos[mask]
        
            atoms = Atoms(numbers = np.concatenate([np.ones_like(h_confs), 8 * np.ones_like(o_confs)]), 
                        positions = np.concatenate([h_pos, o_pos]), 
                        cell = self.cell_info[1:])
            
            atoms.set_array('conf', np.concatenate([h_confs, o_confs]))
        
        else:
            atoms = Atoms(numbers = atoms_types, positions = atoms_pos, cell = self.cell_info[1:])
            atoms.set_array('conf', atoms_confs)
        
        return atoms
        
    def read_cell_info(self, path):
        path = Path(path)
        info_file = next(filter(lambda x: x.suffix in (".txt", ".json", ".xyz"), path.iterdir()), None)
       
        if info_file is None:
            return None
       
        elif info_file.suffix == ".json":
            with open(info_file, "r") as f:
                dic = json.load(f)
                cell_disp = np.zeros((3, ))
                
                if "cell" in dic:
                    cell = np.diag(np.array(dic["cell"]))
                elif "gridA" in dic:
                    cell = np.array([dic["gridA"], dic["gridB"], dic["gridC"]])
                elif "size" in dic:
                    cell = np.diag(np.array([dic['size'][0] * 10, dic['size'][1] * 10, 20.0]))
                
            return np.concatenate([[cell_disp], cell], axis = 0) # shape: (4, 3)
        
        elif info_file.suffix == ".xyz":
            atoms: Atoms = io.read(info_file, format = "extxyz", index = 0) # type: ignore
            cell_disp = atoms.get_celldisp()
            return np.concatenate([[cell_disp], atoms.cell.array], axis = 0) # shape: (4, 3)
        
        elif info_file.suffix == ".txt":
            with open(info_file, "r") as f:
                while not f.readline().startswith("L_exp"):
                    pass
                cell_disp = np.zeros((3,))
                cell = np.diag(np.array(list(map(lambda x: 10 * float(x), f.readline().split()))[::-1] + [20.0]))
                
            return np.concatenate([[cell_disp], cell], axis = 0) # shape: (4, 3)
        
        raise ValueError(f"No valid cell information found in {path}")
   
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
            
        elif self.mode == 'afm+crop':
            afm = self._read_afm_crop(idx)
            atoms = None
            fname = self.fname.stem + f"_{idx}"
                
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