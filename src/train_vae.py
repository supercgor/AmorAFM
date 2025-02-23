import os, time, shutil
import numpy as np
import torch
import utils

from argparse import ArgumentParser
from ase.visualize import view
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt
from pathlib import Path
from utils import box2atom
from dataset import DetectDataset
from network import CVAE3D

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.cVAE import cVAEConfig as Config

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--outdir", type=str, default="outputs/", help="Working directory")

    return parser.parse_args()


class Trainer():

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.outdir = Path(self.cfg.setting.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.cfg.setting.device)
        self.log = utils.get_logger(self.outdir)
        
        self.epoch = 0
        self.iters = 0

        self.model = CVAE3D(**self.cfg.model.params.__dict__).to(self.device)

        self.train_dts = DetectDataset(
            cfg.dataset.train_path,
            mode='label',
            real_size=(25.0, 25.0, 16.0),
            box_size=(25, 25, 8),
            elements=(8, ),
            flipz=True,
            random_zoffset=(-1.5, -0.5),
            random_top_remove_ratio=0.3,
        )

        self.test_dts = DetectDataset(
            cfg.dataset.test_path,
            mode='label',
            real_size=(25.0, 25.0, 16.0),
            box_size=(25, 25, 8),
            elements=(8, ),
            flipz=True,
            random_zoffset=[-1.5, -0.5],
            random_top_remove_ratio=0.3,
        )

        self.train_dtl = DataLoader(self.train_dts,
                                    self.cfg.setting.batch_size,
                                    shuffle=True,
                                    num_workers=self.cfg.setting.num_workers,
                                    pin_memory=self.cfg.setting.pin_memory,
                                    collate_fn=DetectDataset.collate_fn)
        
        self.test_dtl = DataLoader(self.test_dts,
                                   self.cfg.setting.batch_size,
                                   shuffle=True,
                                   num_workers=self.cfg.setting.num_workers,
                                   pin_memory=self.cfg.setting.pin_memory,
                                   collate_fn=DetectDataset.collate_fn)
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.cfg.optimizer.lr,
                                    weight_decay=self.cfg.optimizer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, **self.cfg.scheduler.params.__dict__)
        self.atom_metrics = utils.MetricCollection(
            M=utils.ConfusionMatrix(count_types=("O", ),
                                    real_size=(25.0, 25.0, 16.0),
                                    split=self.cfg.dataset.split,
                                    match_distance=1.0)
            ).to(self.device)

        self.grid_metrics = utils.MetricCollection(
            loss=MeanMetric(),
            grad=MeanMetric(),
            conf=MeanMetric(),
            off=MeanMetric(),
            kl=MeanMetric(),
        ).to(self.device)

        self.save_paths = []
        self.best = np.inf

    def fit(self):
        for epoch in range(1, self.cfg.setting.epoch + 1):
            self.epoch = epoch
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            gm = self.train_one_epoch()
            logstr = f"Train Summary: Epoch: {epoch:2d}, Used time: {(time.time() - epoch_start_time)/60:4.1f} mins, last loss: {gm['loss']:.2e}"

            gm, atom_metric = self.test_one_epoch()
            loss = gm['loss']
            M = atom_metric['M']
            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if loss < self.best else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {M[:, :,3].mean():.2f} | AR: {M[:, :,4].mean():.2f} | ACC: {M[:, :,5].mean():.2f} | SUC: {M[:, :,6].mean():.2f}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {M[0, i,3]:.2f} | AR: {M[0, i,4]:.2f} | ACC: {M[0, i,5]:.2f} | SUC: {M[0, i,6]:.2f}\nTP: {M[0, i,0]:10.0f} | FP: {M[0, i,1]:10.0f} | FN: {M[0, i,2]:10.0f}"

            self.save_model(loss)
            self.log.info(logstr)

    def train_one_epoch(self):
        self.grid_metrics.reset()
        self.atom_metrics.reset()
        self.model.train()

        for i, (filenames, _, targs, atoms) in enumerate(self.train_dtl):
            targs = targs.to(self.device, non_blocking=True)
            self.opt.zero_grad()

            out, latents, conds = self.model(targs)

            loss, loss_values = self.model.compute_loss(
                out, targs, latents, conds)
            loss.backward()

            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.cfg.optimizer.clip_grad,
                                                  error_if_nonfinite=True)
            self.opt.step()

            out_atoms = box2atom(out, (25, 25, 16), 0.5, 2.0)

            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(loss=loss,
                                     grad=grad,
                                     conf=loss_values['conf'],
                                     off=loss_values['offset'],
                                     kl=loss_values['vae'])

            self.iters += 1

            if i % self.cfg.setting.log_every == 0:
                # print(out.shape)
                # view(atoms, block=True)
                out = out[0, ..., 0].detach().cpu()  # X Y Z
                out = out.permute(2, 1, 0)[:, None]  # Z 1 Y X
                out = make_grid(out, nrow=8, pad_value=0.3)[[0]]

                targs = targs[0, ..., 0].detach().cpu()
                targs = targs.permute(2, 1, 0)[:, None]
                targs = make_grid(targs, nrow=8, pad_value=0.3)[[0]]
                # print(targs.shape, out.shape)
                fig = plt.figure(figsize=(16, 8))
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(torch.cat([targs, targs, out], 0).permute(1, 2, 0))
                fig.savefig(self.outdir/ f"points_{self.epoch:02d}.png")
                plt.close(fig)
                # save_image(torch.cat([targs, targs, out], 0), f"{self.work_dir}/train_{epoch}.png")

                self.log.info(f"E[{self.epoch:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.train_dtl):5d}], L{loss:.2e}, G{grad:.2e}")
                utils.log_to_csv(self.outdir / f"train.csv",
                                 total=self.iters,
                                 epoch=self.epoch,
                                 iter=i,
                                 **self.grid_metrics.compute())
                self.grid_metrics.reset()

            if self.cfg.setting.debug and i > 100:
                break

        self.scheduler.step()

        return self.grid_metrics.compute()

    @torch.no_grad()
    def test_one_epoch(self):
        self.grid_metrics.reset()
        self.atom_metrics.reset()
        self.model.eval()
        for i, (filenames, _, targs, atoms) in enumerate(self.test_dtl):
            targs = targs.to(self.device, non_blocking=True)

            out, latents, conds = self.model(targs)

            loss, loss_values = self.model.compute_loss(out, targs, latents, conds)

            out_atoms = box2atom(out, (25, 25, 16), 0.5, 2.0)

            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(loss=loss,
                                     conf=loss_values['conf'],
                                     off=loss_values['offset'],
                                     kl=loss_values['vae'])

            if i % self.cfg.setting.log_every == 0:
                self.log.info(f"E[{self.epoch:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.test_dtl):5d}], L{loss:.2e}")

            if self.cfg.setting.debug and i > 100:
                break

        return self.grid_metrics.compute(), self.atom_metrics.compute()

    def save_model(self, metric = None):
        if metric is None or metric < self.best:
            self.best = metric
            path = self.outdir / f"EP{self.epoch}_L{metric:.2f}.pkl"
            if len(self.save_paths) >= self.cfg.setting.max_save:
                os.remove(self.save_paths.pop(0))
            torch.save(self.model.state_dict(), path)
            self.save_paths.append(path)

    def load_model(self):
        if self.cfg.model.checkpoint != "":
            params = torch.load(self.cfg.model.checkpoint, map_location=self.device, weights_only=True)
            mismatch = self.model.load_state_dict(params, strict = False)
            self.log.info(f"Load model parameters from {self.cfg.model.checkpoint}")
            if not mismatch.missing_keys and not mismatch.unexpected_keys:
                self.log.info("Model loaded successfully")
            else:
                if mismatch.missing_keys:
                    self.log.info(f"Missing keys: {mismatch.missing_keys}")
                if mismatch.unexpected_keys:
                    self.log.info(f"Unexpected keys: {mismatch.unexpected_keys}")       
        else:
            self.log.info("Start a new model")
          
    def log_status(self):
        self.log.info(f"Output directory: {self.cfg.setting.outdir}")
        self.log.info(f"Using devices: {next(self.model.parameters()).device}")
        self.log.info(f"Precison: {torch.get_default_dtype()}")
        self.log.info(f"Model parameters: {sum([p.numel() for p in self.model.parameters()])}")
    
def main():
    args = get_parser()

    cfg = Config()

    outdir = Path(args.outdir) / f"{time.strftime('%Y%m%d-%H%M%S')}-cycleGAN"
    
    cfg.setting.device = args.device
    cfg.setting.outdir = str(outdir)
    cfg.setting.debug = args.debug
    
    if cfg.setting.debug:
        cfg.setting.log_every = 1
        cfg.setting.batch_size = 2
        cfg.setting.max_save = 1
        cfg.setting.epoch = 5
    
    try:
        start_time = time.time()
        trainer = Trainer(cfg)
        trainer.fit()
        
    except Exception as e:
        if not args.debug and time.time() - start_time < 300:
            shutil.rmtree(cfg.setting.outdir)
        raise e


if __name__ == "__main__":
    main()
