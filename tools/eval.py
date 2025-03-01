import sys
import torch
import numpy as np
import argparse

from ase import io, Atoms
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution as de
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.const import Ih_positions, Ih_cell
from src.utils import box2atom, water_solver, vec2box, write_water_data
from src.network import UNetND, CVAE3D
from src.dataset import DetectDataset
from configs.cVAE import cVAEConfig
from configs.detect import DetectConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description='Transform the AFM images to the crystal structure')
    parser.add_argument('-i', '--path', type=str,
                        help='Path to the AFM images or structure', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='Output path', default='outputs/')
    parser.add_argument('-f', '--output-format', type=str,
                        help='Output format', choices=['xyz', 'data'], default='xyz')

    parser.add_argument('--device', type=str, help='Device to use', 
                        choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--detect-model', type=str, help='Path to the pretrain model',
                        default="dataset/pretrain-det.pkl")
    parser.add_argument('--match-model', type=str, help='Path to the cVAE model',
                        default="dataset/pretrain-vae.pkl")
    parser.add_argument('--save-immediate', action='store_true',
                        help='Save the intermediate results')

    parser.add_argument('-d', '--detect-only',
                        action='store_true', help='Only detect the AFM images')
    parser.add_argument('-m', '--match-only', action='store_true',
                        help='Only match the crystal part')
    parser.add_argument('--z-ref', type=float, default=1.0,
                        help='The reference z value for the crystal part')
    parser.add_argument('--z-padding', type=float, default=20.0,
                        help='The padding value after combining the crystal part and amorphous part in z direction')
    return parser.parse_args()

def find_best(atoms, s_tag=0.5, n=10):
    atoms = atoms.copy()
    atoms_cell = atoms.cell.array
    atoms.positions[:, :2] -= (atoms_cell[0, 0] / 2, atoms_cell[1, 1] / 2)
    pos = Ih_positions[::3] % np.diag(Ih_cell)
    atree = KDTree(pos, boxsize=np.diag(Ih_cell))
    a, z = Ih_cell[0, 0] / 3, Ih_cell[2, 2]
    bounds = [(-0.5 * a,              a),
              (-0.5 * np.sqrt(3) * a, 0.5 * np.sqrt(3) * a),
              (-z / 4,                z/4),
              (-np.pi/3,              np.pi/3)]

    best_score = np.inf

    for i in range(n):
        def loss_fn(params):
            r = R.from_euler('z', params[3])
            pos = r.apply(atoms.positions) + params[:3]
            distances, _ = atree.query(pos)
            return np.partition(distances, 100)[:100].mean()

        result = de(loss_fn, bounds, strategy='best1bin',
                    popsize=200, recombination=0.9)
        if result.fun < best_score:
            param = result.x
            best_score = result.fun

        if i >= 5 and result.fun < s_tag:
            break

    r = R.from_euler('z', param[3])
    atoms.positions = r.apply(atoms.positions)
    atoms.positions += param[:3]
    return atoms, param, best_score


def main(args):
    path = Path(args.path)
    det_model_path = Path(args.detect_model)
    mat_model_path = Path(args.match_model)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amorphous_part: Atoms | None = None
    print(f"Processing: {path}")
    if args.detect_only or not args.match_only:
        print("Transforming afm to atoms...")
        print("Loading detect model...")

        cfg = DetectConfig()
        net = UNetND(**cfg.model.params.__dict__)

        net.load_state_dict(torch.load(det_model_path, map_location=device), strict=True)
        net.to(device).eval().requires_grad_(False)

        train_dts = DetectDataset(path,
                                mode='afm+crop',
                                num_images=10,
                                image_size=(100, 100),
                                image_split=None,
                                real_size=cfg.dataset.real_size,
                                box_size=(32, 32, 4),
                                random_transform=False,
                                normalize=False
                                )

        pre_atoms = []

        print("Detecting AFM images...")

        for i in range(len(train_dts)):
            name, afm, _, _ = train_dts[i]
            assert afm is not None

            with torch.no_grad():
                pred = net(afm[None].to(device)).cpu().numpy()

            atoms = box2atom(pred[0],
                            cfg.dataset.real_size,
                            0.5,
                            cutoff=(1.036, 0.7392),
                            nms=cfg.dataset.nms)

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(np.rot90(afm[0, 0].numpy()), extent = (0, 25, 0, 25))
            # ax.scatter(atoms.positions[:, 0], atoms.positions[:, 1], c='r', s=1)
            # plt.show()
            # plt.close()
            pre_atoms.append(atoms)

        print("Combining AFM images...")

        if args.save_immediate:
            raw_atoms = train_dts.combine_label_crop(pre_atoms, nms=False)
            io.write(out_path / f"{path.stem}_afm.xyz", raw_atoms)

        pre_atoms_combine = train_dts.combine_label_crop(pre_atoms)

        amorphous_part = water_solver(pre_atoms_combine)
        
        if args.output_format == 'xyz':
            io.write(out_path / f"{path.stem}_water.xyz", amorphous_part) # type: ignore
        elif args.output_format == 'data':
            write_water_data(out_path / f"{path.stem}_water.data", amorphous_part.positions, amorphous_part.cell.array) 
        
    if args.match_only or not args.detect_only:
        print("Generating crystal part...")
        if amorphous_part is None:
            amorphous_part = io.read(path, index=0) # type: ignore
            
        print("Loading cVAE model...")

        cfg = cVAEConfig()
        net = CVAE3D(**cfg.model.params.__dict__)

        net.load_state_dict(torch.load(mat_model_path, map_location=device), strict=True)
        net.to(device).eval().requires_grad_(False)
        
        o_combine_atoms: Atoms = amorphous_part[amorphous_part.numbers == 8] # type: ignore
        o_pos = o_combine_atoms.positions

        o_cell = np.diag(o_combine_atoms.cell.array)
        real_size = np.array([25.0, 25.0, 4.0])
        o_center = np.array([o_cell[0] / 2, o_cell[1] / 2, -args.z_ref])

        o_pos += np.array([real_size[0] / 2 - o_cell[0] / 2,
                           real_size[1] / 2 - o_cell[1] / 2,
                           o_center[2]])

        o_pos = o_pos[np.all(o_pos > 0, axis=1) &
                    np.all(o_pos < real_size, axis=1)]
        o_pos /= real_size

        box = vec2box(o_pos, box_size=[25, 25, 2])
        box = torch.as_tensor(box, dtype=torch.float32, device=device)

        print("Sampling crystal part...")

        out = net.conditional_sample(box[None], resample=False).cpu().numpy()[0]

        out_atoms = box2atom(out, cfg.dataset.real_size, 0.5, cutoff=2.0, nms=True)
        
        if args.save_immediate:
            io.write(out_path / f"{path.stem}_match.xyz", out_atoms)

        
        out_crystal_part: Atoms = out_atoms[out_atoms.positions[:, 2] < 12] # type: ignore
        out_crystal_part.positions[:, 2] += -12

        print("Matching crystal part...")

        out_atoms, best_param, best_score = find_best(
            out_crystal_part, s_tag=70.0, n=10)

        print(f"Matching score: {best_score}, best param: {best_param}")

        print("Combining crystal part and amorphous part...")

        ang = best_param[3]

        cell_x = o_cell[0] * np.abs(np.cos(ang)) + o_cell[1] * np.abs(np.sin(ang))
        cell_y = o_cell[0] * np.abs(np.sin(ang)) + o_cell[1] * np.abs(np.cos(ang))

        cell_x = np.rint(cell_x / Ih_cell[0, 0]) + 1
        cell_y = np.rint(cell_y / Ih_cell[1, 1]) + 1
        cell_z = 3

        grids = np.stack(np.meshgrid(np.arange(cell_x),
                                     np.arange(cell_y),
                                     np.arange(cell_z)), axis=-1).reshape(-1, 3)  # N x 3

        crystal_part_pos = (Ih_positions[None] + (grids @ Ih_cell)[:, None]).reshape(-1, 3)
        crystal_part_cell = Ih_cell * np.array([cell_x, cell_y, cell_z])
        crystal_part_num = np.tile([8, 1, 1], 16 * grids.shape[0])
        crystal_part = Atoms(positions=crystal_part_pos, cell=crystal_part_cell, numbers=crystal_part_num)
        crystal_part.set_array('id', np.arange(len(crystal_part)))
        
        amorphous_part.positions[:, :2] -= (o_cell[0] / 2, o_cell[1] / 2)
        amorphous_part.positions = R.from_euler('z', ang).apply(amorphous_part.positions)
        amorphous_part.positions += (best_param[0] + (cell_x // 2) * Ih_cell[0, 0],
                                     best_param[1] + (cell_y // 2) *  Ih_cell[1, 1],
                                     best_param[2] + cell_z * Ih_cell[2, 2] - args.z_ref)
        
        amorphous_part.arrays['id'] += len(crystal_part)
        
        results = crystal_part + amorphous_part
        results.cell[2, 2] += args.z_padding
        
        if args.output_format == 'xyz':
            io.write(out_path / f"{path.stem}_combine.xyz", results)
        elif args.output_format == 'data':
            write_water_data(out_path / f"{path.stem}_combine.data", results.positions, results.cell.array)
        
        print(f"Done! Results are saved in {out_path}")
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
