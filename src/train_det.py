import os, sys, time, shutil
import numpy as np
import time
import torch
import utils

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.detect import DetectConfig as Config
from src.network import UNetND, CycleGAN
from src.dataset import DetectDataset
from src.utils import box2atom, plot_preditions

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode", default=True)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--outdir", type=str, default="outputs/", help="Working directory")
    return parser.parse_args()

class Trainer():
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(self.cfg.setting.device)
        self.outdir = Path(self.cfg.setting.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log = utils.get_logger("train", self.outdir)
        self.iters = 0
        self.epoch = 0

        self.model = UNetND(**self.cfg.model.params.__dict__).to(self.device)
      
        if cfg.tune_model.checkpoint != "":
            self.tune_model = CycleGAN(**self.cfg.tune_model.params.__dict__)            
            self.tune_model.requires_grad_(False).eval().to(self.device)
            
        else:
            self.tune_model = None
        
        self.load_model()
        
        self.train_dts = DetectDataset(cfg.dataset.train_path,
                                       mode='afm+label',
                                       num_images=cfg.dataset.num_images,
                                       image_size=cfg.dataset.image_size,
                                       image_split=cfg.dataset.image_split,
                                       real_size=self.cfg.dataset.real_size,
                                       box_size=self.cfg.dataset.box_size,
                                       random_transform=True,
                                       random_noisy=0.1,
                                       random_cutout=True,
                                       random_jitter=True,
                                       random_blur=True,
                                       random_shift=True,
                                       random_flipx=False,
                                       random_flipy=False)

        self.test_dts = DetectDataset(cfg.dataset.test_path,
                                      mode='afm+label',
                                      num_images=self.cfg.dataset.num_images,
                                      image_size=self.cfg.dataset.image_size,
                                      image_split=self.cfg.dataset.image_split,
                                      real_size=self.cfg.dataset.real_size,
                                      box_size=self.cfg.dataset.box_size,
                                      random_transform=True,
                                      random_noisy=0.1,
                                      random_cutout=True,
                                      random_jitter=True,
                                      random_blur=True,
                                      random_shift=True)

        collate_fn = self.train_dts.collate_fn

        if self.cfg.setting.debug:
            self.train_dts = torch.utils.data.Subset(self.train_dts, range(100))
            self.test_dts = torch.utils.data.Subset(self.test_dts, range(20))
        
        self.train_dtl = DataLoader(self.train_dts,
                                    cfg.setting.batch_size,
                                    True,
                                    num_workers=cfg.setting.num_workers,
                                    pin_memory=cfg.setting.pin_memory,
                                    collate_fn=collate_fn)

        self.test_dtl = DataLoader(self.test_dts,
                                   cfg.setting.batch_size,
                                   False,
                                   num_workers=cfg.setting.num_workers,
                                   pin_memory=cfg.setting.pin_memory,
                                   collate_fn=collate_fn)

        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.cfg.optimizer.lr,
                                    weight_decay=self.cfg.optimizer.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
                                     self.opt,
                                     **self.cfg.scheduler.params.__dict__)

        self.atom_metrics = utils.MetricCollection(
            M=utils.ConfusionMatrix(
                real_size=cfg.dataset.real_size,
                split=self.cfg.dataset.split,
                match_distance=1.0)
        ).to(self.device)

        self.grid_metrics = utils.MetricCollection(
            loss=MeanMetric(),
            grad=MeanMetric(),
            conf=MeanMetric(),
            xy=MeanMetric(),
            z=MeanMetric(),
        ).to(self.device)

        self.log_status()
        
        self.save_paths = []
        self.best = np.inf

    def fit(self):
        for epoch in range(1, self.cfg.setting.epoch + 1):
            self.epoch = epoch
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            gm = self.train_one_epoch()
            self.log.info(f"Train Summary: Epoch: {epoch:2d}, Used time: {(time.time() - epoch_start_time)/60:4.1f} mins, last loss: {gm['loss']:.2e}")

            gm, atom_metric = self.test_one_epoch()
            loss = gm['loss']
            M = atom_metric['M']

            utils.log_to_csv(self.outdir / "test.csv",
                             total=self.iters,
                             epoch=epoch,
                             **gm)
            
            self.log.info(f"Test Summary, Epoch {self.epoch:2d}")
            self.log.info(f"Loss: {loss:.2e}, Used time: {(time.time() - epoch_start_time)/60:4.1f} mins")
            self.log.info(f"{'Model saved' if loss < self.best else 'Model not saved'}")
            self.log.info(f"Overall metrics")
            self.log.info(f"AP: {M[:, :, 3].mean():10.2f}, AR: {M[:, :,4].mean():10.2f}, ACC: {M[:, :,5].mean():9.2f}")
            self.log.info(f"TP: {M[:, :, 0].mean():10.0f}, FP: {M[:, :, 1].mean():10.0f}, FN: {M[:, :, 2].mean():10.0f}")
            self.log.info(f"SUC: {M[:, :,6].mean():9.2f}, F1: {(M[:, :,3] * M[:, :,4] * 2 / (M[:, :,3] + M[:, :,4])).mean():10.2f}")
            for elem in range(M.shape[0]):
                elem_name = "O" if elem == 0 else "H"
                self.log.info(f"{elem_name} metrics")
                self.log.info(f"AP: {M[elem, :,3].mean():10.2f}, AR: {M[elem, :,4].mean():10.2f}, ACC: {M[elem, :,5].mean():9.2f}")
                self.log.info(f"TP: {M[elem, :,0].mean():10.0f}, FP: {M[elem, :, 1].mean():10.0f}, FN: {M[elem, :,2].mean():10.0f}")
                self.log.info(f"SUC: {M[elem, :,6].mean():9.2f}, F1: {(M[elem, :,3] * M[elem, :,4] * 2 / (M[elem, :,3] + M[elem, :,4])).mean():10.2f}")
                for i in range(len(self.cfg.dataset.split) - 1):
                    low, high = self.cfg.dataset.split[i: i+2]
                    self.log.info(f"({low:.1f}-{high:.1f}A) AP: {M[elem,i,3]:10.2f}, AR: {M[elem,i,4]:10.2f}, ACC: {M[elem,i,5]:9.2f}")
                    self.log.info(f"({low:.1f}-{high:.1f}A) TP: {M[elem,i,0]:10.0f}, FP: {M[elem,i,1]:10.0f}, FN: {M[elem,i,2]:10.0f}")
                    self.log.info(f"({low:.1f}-{high:.1f}A) SUC: {M[elem,i,6]:9.2f}, F1: {(M[elem, i,3] * M[elem, i,4] * 2 / (M[elem, i,3] + M[elem, i,4])).mean():10.2f}")

            self.save_model(loss)

    def train_one_epoch(self):
        self.grid_metrics.reset()
        self.atom_metrics.reset()
        self.model.train()

        for i, (filenames, inps, targs, atoms) in enumerate(self.train_dtl):
            inps = inps.to(self.device, non_blocking=True)
            targs = targs.to(self.device, non_blocking=True)
            self.opt.zero_grad()

            if self.tune_model is not None:
                w = torch.rand(inps.shape[0], 1, 1, 1, 1, device=inps.device)
                with torch.no_grad():
                    inps = self.tune_model.to_B(inps) * w + inps * (1 - w)

            preds = self.model(inps)

            loss, loss_values = self.model.compute_loss(preds, targs)
            loss.backward()

            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.cfg.optimizer.clip_grad,
                                                  error_if_nonfinite=False)
            self.opt.step()
            out_atoms = box2atom(preds.detach().cpu().numpy(),
                                 self.cfg.dataset.real_size,
                                 0.5,
                                 cutoff=(1.036, 0.7392),
                                 mode='OH',
                                 nms=self.cfg.dataset.nms)

            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(loss=loss,
                                     grad=grad,
                                     conf=loss_values['conf'],
                                     xy=loss_values['xy'],
                                     z=loss_values['z'])

            self.iters += 1

            if i % self.cfg.setting.log_every == 0:
                savedir = self.outdir / f"Epoch{self.epoch:02d}"
                savedir.mkdir(exist_ok=True)
                with torch.no_grad():
                    plot_preditions(savedir / f"{i:06d}.png", inps[0].detach().cpu().numpy(), out_atoms[0], atoms[0], name = filenames[0])
                    
                    
                    self.log.info(
                        f"E[{self.epoch:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.train_dtl):5d}], L{loss:.2e}, G{grad:.2e}"
                    )
                    
                    utils.log_to_csv(self.outdir / "train.csv",
                                     total=self.iters,
                                     epoch=self.epoch,
                                     iter=i,
                                     **self.grid_metrics.compute())
                self.grid_metrics.reset()

        self.scheduler.step()

        return self.grid_metrics.compute()

    @torch.no_grad()
    def test_one_epoch(self):
        self.grid_metrics.reset()
        self.atom_metrics.reset()
        self.model.eval()
        for i, (filenames, inps, targs, atoms) in enumerate(self.test_dtl):
            inps = inps.to(self.device, non_blocking=True)
            targs = targs.to(self.device, non_blocking=True)

            if self.tune_model is not None:
                w = torch.rand(inps.shape[0], 1, 1, 1, 1, device=inps.device)
                with torch.no_grad():
                    inps = self.tune_model.to_B(inps) * w + inps * (1 - w)

            preds = self.model(inps)
            loss, loss_values = self.model.compute_loss(preds, targs)

            out_atoms = box2atom(preds.detach().cpu().numpy(),
                                 tuple(self.cfg.dataset.real_size),
                                 0.5,
                                 cutoff=(1.036, 0.7392),
                                 mode='OH',
                                 nms=self.cfg.dataset.nms,
                                 num_workers=self.cfg.setting.num_workers)

            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(loss=loss,
                                     conf=loss_values['conf'],
                                     xy=loss_values['xy'],
                                     z=loss_values['z'])

            if i % self.cfg.setting.log_every == 0:
                self.log.info(
                    f"E[{self.epoch:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.test_dtl):5d}], L{loss:.2e}"
                )

        return self.grid_metrics.compute(), self.atom_metrics.compute()

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
            
        if self.tune_model is not None and self.cfg.tune_model.checkpoint != "":
            params = torch.load(self.cfg.tune_model.checkpoint, map_location=self.device, weights_only=True)
            mismatch = self.tune_model.load_state_dict(params, strict = False)
            self.log.info(f"Load tune model parameters from {self.cfg.tune_model.checkpoint}")
            if not mismatch.missing_keys and not mismatch.unexpected_keys:
                self.log.info("Tune model loaded successfully")
            else:
                if mismatch.missing_keys:
                    self.log.info(f"Missing keys: {mismatch.missing_keys}")
                if mismatch.unexpected_keys:
                    self.log.info(f"Unexpected keys: {mismatch.unexpected_keys}")
    
    def log_status(self):
        self.log.info(f"Output directory: {self.cfg.setting.outdir}")
        self.log.info(f"Using devices: {next(self.model.parameters()).device}")
        self.log.info(f"Precison: {torch.get_default_dtype()}")
        self.log.info(f"Model parameters: {sum([p.numel() for p in self.model.parameters()])}")
        if self.tune_model is not None:
            self.log.info(f"Tune model parameters: {sum([p.numel() for p in self.tune_model.parameters()])}")
        
    
    def save_model(self, metric):
        # save model
        if metric is None or metric < self.best:
            self.best = metric
            path = self.outdir / f"DETECT_E{self.epoch:02d}_L{metric:.3e}.pkl"
            if len(self.save_paths) >= self.cfg.setting.max_save:
                os.remove(self.save_paths.pop(0))
            torch.save(self.model.state_dict(), path)
            self.save_paths.append(path)

def main():
    args = get_parser()
    
    cfg = Config()
    
    outdir = Path(args.outdir) / f"{time.strftime('%Y%m%d-%H%M%S')}-detect"
    
    cfg.setting.device = args.device
    cfg.setting.outdir = str(outdir)
    cfg.setting.debug = args.debug
    
    if cfg.setting.debug:
        cfg.setting.num_workers = 0
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