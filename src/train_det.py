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
from network import UNetND, CycleGAN
from dataset import DetectDataset
from utils import box2atom, view
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.detect import DetectConfig as Config

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
                                       box_size=(32, 32, 4),
                                       random_transform=True,
                                       random_noisy=0.3,
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
                                      box_size=(32, 32, 4),
                                      random_transform=True,
                                      random_noisy=0.3,
                                      random_cutout=True,
                                      random_jitter=True,
                                      random_blur=True,
                                      random_shift=True)

        collate_fn = self.train_dts.collate_fn

        if self.cfg.setting.debug:
            self.train_dts = torch.utils.data.Subset(self.train_dts, range(10))
            self.test_dts = torch.utils.data.Subset(self.test_dts, range(10))
        
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
                real_size=tuple(cfg.dataset.real_size),
                split=self.cfg.dataset.split,
                match_distance=1.0)).to(self.device)

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
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            gm = self.train_one_epoch(epoch,
                                      log_every=self.cfg.setting.log_every)
            logstr = f"Train Summary: Epoch: {epoch:2d}, Used time: {(time.time() - epoch_start_time)/60:4.1f} mins, last loss: {gm['loss']:.2e}"

            gm, atom_metric = self.test_one_epoch(
                epoch, log_every=self.cfg.setting.log_every)
            loss = gm['loss']
            M = atom_metric['M']

            utils.log_to_csv(self.outdir / "test.csv",
                             total=self.iters,
                             epoch=epoch,
                             **gm)

            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if loss < self.best else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {M[:, :,3].mean():.2f} | AR: {M[:, :,4].mean():.2f} | ACC: {M[:, :,5].mean():.2f} | SUC: {M[:, :,6].mean():.2f} | F1: {(M[:, :,3] * M[:, :,4] * 2 / (M[:, :,3] + M[:, :,4])).mean():.2f}"
            for elem in range(M.shape[0]):
                logstr += f"\n({'O' if elem == 0 else 'H'}) AP: {M[elem, :,3].mean():.2f} | AR: {M[elem, :,4].mean():.2f} | ACC: {M[elem, :,5].mean():.2f} | SUC: {M[elem, :,6].mean():.2f} | F1: {(M[elem, :,3] * M[elem, :,4] * 2 / (M[elem, :,3] + M[elem, :,4])).mean():.2f}"
                for i, (low, high) in enumerate(
                        zip(self.cfg.dataset.split[:-1],
                            self.cfg.dataset.split[1:])):
                    logstr += f"\n({low:.1f}-{high:.1f}A) AP: {M[elem,i,3]:.2f} | AR: {M[elem,i,4]:.2f} | ACC: {M[elem,i,5]:.2f} | SUC: {M[elem,i,6]:.2f}\nTP: {M[elem,i,0]:10.0f} | FP: {M[elem,i,1]:10.0f} | FN: {M[elem,i,2]:10.0f}"

            self.save_model(epoch, loss)
            self.log.info(logstr)

    def train_one_epoch(self, epoch, log_every: int = 100):
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
                                 tuple(self.cfg.dataset.real_size),
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

            if i % log_every == 0:
                with torch.no_grad():
                    plot_inp, plot_targ, plot_pred, plot_atoms = inps[0], targs[0], preds[0], out_atoms[0]
                    images = view.afm_points_to_grid(plot_inp, plot_pred)
                    save_image(images, self.outdir / "test.png")
                    save_image(images[0], self.outdir / "omap.png")
                    save_image(images[1], self.outdir / "pmap.png")
                    # print(preds[0,...,0].max(), targs[0,...,0].max())
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111)
                    ax.imshow(inps[0, :, :, 3].detach().cpu().numpy().transpose(1, 0, 2),
                              origin='lower',
                              cmap='gray')
                    scale_x = inps.shape[1] / self.cfg.dataset.real_size[0]
                    scale_y = inps.shape[2] / self.cfg.dataset.real_size[1]
                    
                    ax.scatter(out_atoms[0].get_positions()[:, 0] * scale_x, # type: ignore
                               out_atoms[0].get_positions()[:, 1] * scale_y, # type: ignore
                               c='r')
                    ax.scatter(atoms[0].positions[:, 0] * scale_x,
                               atoms[0].positions[:, 1] * scale_y,
                               c='b')
                    fig.savefig(self.outdir / "atom.png")
                    plt.close(fig)
                    self.log.info(
                        f"E[{epoch:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.train_dtl):5d}], L{loss:.2e}, G{grad:.2e}"
                    )
                    
                    utils.log_to_csv(self.outdir / "train.csv",
                                     total=self.iters,
                                     epoch=epoch,
                                     iter=i,
                                     **self.grid_metrics.compute())
                self.grid_metrics.reset()

            if self.cfg.setting.debug and i > 10:
                break

        self.scheduler.step()

        return self.grid_metrics.compute()

    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 100):
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

            if i % log_every == 0:
                self.log.info(
                    f"E[{epoch:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.test_dtl):5d}], L{loss:.2e}"
                )

            if self.cfg.setting.debug and i > 10:
                break

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
        
    
    def save_model(self, epoch, metric):
        # save model
        if metric is None or metric < self.best:
            self.best = metric
            path = self.outdir / f"unetv3_CP{epoch:02d}_L{metric:.4f}.pkl"
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