import os
import sys
import time
import shutil
import numpy as np
import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.image import FrechetInceptionDistance as FID
from matplotlib import pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.cycleGAN import CycleGANConfig as Config
from src.utils import ItLoader, MetricCollection, log_to_csv, get_logger
from src.dataset import DetectDataset
from src.network import CycleGAN

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
        self.log = get_logger(self.outdir)

        self.epoch = 0
        self.iters = 0
        self.max_iters = 10000

        self.model = CycleGAN(**cfg.model.params.__dict__).to(self.device)

        self.load_model()
        

        self.A_dts = DetectDataset(
            cfg.dataset.source_path,
            mode='afm',
            num_images=self.cfg.dataset.num_images,
            image_size=self.cfg.dataset.image_size,
            image_split=self.cfg.dataset.image_split,
            real_size=self.cfg.dataset.real_size,
            random_blur=0.0,
            random_cutout=True,
            random_jitter=False,
            random_noisy=0.05,
            random_shift=True,
        )

        # self.A_dts.key_filter(lambda x: 'icehup' in x)

        self.B_dts = DetectDataset(
            cfg.dataset.target_path,
            mode='afm',
            num_images=self.cfg.dataset.num_images,
            image_size=self.cfg.dataset.image_size,
            image_split=self.cfg.dataset.image_split,
            real_size=self.cfg.dataset.real_size,
            random_blur=0.0,
            random_cutout=False,
            random_jitter=False,
            random_noisy=0.01,
            random_shift=False,
        )

        self.A_dtl_train = DataLoader(
            self.A_dts,
            batch_size=self.cfg.setting.batch_size,
            sampler=ItLoader(self.A_dts, self.cfg.setting.max_iters, True),
            num_workers=self.cfg.setting.num_workers,
            pin_memory=self.cfg.setting.pin_memory,
            collate_fn=DetectDataset.collate_fn,
        )

        self.B_dtl_train = DataLoader(
            self.B_dts,
            batch_size=self.cfg.setting.batch_size,
            sampler=ItLoader(self.B_dts, self.cfg.setting.max_iters, True),
            num_workers=self.cfg.setting.num_workers,
            pin_memory=self.cfg.setting.pin_memory,
            collate_fn=DetectDataset.collate_fn,
        )

        self.A_dtl_test = DataLoader(
            self.A_dts,
            batch_size=self.cfg.setting.batch_size,
            sampler=ItLoader(self.A_dts, 1600, True),
            num_workers=self.cfg.setting.num_workers,
            pin_memory=self.cfg.setting.pin_memory,
            collate_fn=DetectDataset.collate_fn,
        )

        self.B_dtl_test = DataLoader(
            self.B_dts,
            batch_size=self.cfg.setting.batch_size,
            sampler=ItLoader(self.B_dts, 1600, True),
            num_workers=self.cfg.setting.num_workers,
            pin_memory=self.cfg.setting.pin_memory,
            collate_fn=DetectDataset.collate_fn,
        )

        self.G_opt = torch.optim.Adam(self.model.G_params,
                                      lr=self.cfg.optimizer.lr,
                                      betas=(0.5, 0.999))

        self.D_opt = torch.optim.Adam(self.model.D_params,
                                      lr=self.cfg.optimizer.lr / 4,
                                      betas=(0.5, 0.999))

        self.G_scheduler = torch.optim.lr_scheduler.StepLR(self.G_opt, **self.cfg.scheduler.params.__dict__)

        self.D_scheduler = torch.optim.lr_scheduler.StepLR(self.D_opt, **self.cfg.scheduler.params.__dict__)
        
        self.gen_metrics = MetricCollection(
            G_to_A_loss=MeanMetric(),
            G_to_A_grad=MeanMetric(),
            G_to_A_cls=MeanMetric(),
            G_to_A_cyc=MeanMetric(),
            G_to_A_idt=MeanMetric(),
            G_to_B_loss=MeanMetric(),
            G_to_B_grad=MeanMetric(),
            G_to_B_cls=MeanMetric(),
            G_to_B_cyc=MeanMetric(),
            G_to_B_idt=MeanMetric(),
        ).to(self.device)

        self.disc_metrics = MetricCollection(
            D_A_loss=MeanMetric(),
            D_A_grad=MeanMetric(),
            D_B_loss=MeanMetric(),
            D_B_grad=MeanMetric(),
        ).to(self.device)

        self.source_fid = FID(feature=64,  input_img_size=(1, 100, 100)).to(self.device)
        self.target_fid = FID(feature=64,  input_img_size=(1, 100, 100)).to(self.device)

        self.save_paths = []
        self.best = np.inf

    def fit(self):
        for epoch in range(1, self.cfg.setting.epoch + 1):
            self.epoch = epoch
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            gm, dm = self.train_one_epoch()
            logstr = f"Epoch {epoch:5d}/{self.cfg.setting.epoch} | Time: {time.time() - epoch_start_time:.2f} | "

            source_fid, target_fid = self.test_one_epoch()

            logstr += f"Source FID: {source_fid:.4f} | Target FID: {target_fid:.4f} | "

            self.save_model()
            self.log.info(logstr)

    def train_one_epoch(self):
        for i, ((source_filenames, source_afm, _, _),
                (target_filenames, target_afm, _,
                 _)) in enumerate(zip(self.A_dtl_train, self.B_dtl_train)):
            source_afm = source_afm.to(self.device, non_blocking=True)
            target_afm = target_afm.to(self.device, non_blocking=True)

            if i % 2:  # Generator
                self.G_opt.zero_grad()

                loss, loss_values = self.model.forward_gen(source_afm, target_afm)

                G_to_A_grad = torch.nn.utils.clip_grad_norm_(self.model.G_to_A.parameters(), 10)
                G_to_B_grad = torch.nn.utils.clip_grad_norm_(self.model.G_to_B.parameters(), 10)

                self.G_opt.step()

                self.gen_metrics.update(G_to_A_grad=G_to_A_grad,
                                        G_to_B_grad=G_to_B_grad,
                                        **loss_values)

            else:  # Discriminator
                self.D_opt.zero_grad()

                loss, loss_values = self.model.forward_disc(
                    source_afm, target_afm)

                D_A_grad = torch.nn.utils.clip_grad_norm_(
                    self.model.D_A.parameters(), 10)
                D_B_grad = torch.nn.utils.clip_grad_norm_(
                    self.model.D_B.parameters(), 10)

                self.D_opt.step()

                self.disc_metrics.update(D_A_grad=D_A_grad,
                                         D_B_grad=D_B_grad,
                                         **loss_values)

            self.iters += 1

            if i > 0 and i % self.cfg.setting.log_every == 0:
                log_to_csv(self.outdir / "train_cyc.csv",
                                 total=self.iters,
                                 **self.gen_metrics.compute(),
                                 **self.disc_metrics.compute())

                fig = plt.figure(figsize=(15, 10))
                fig.suptitle(f"{source_filenames[0]} <-> {target_filenames[0]}")
                
                savedir = self.outdir / f"Epoch{self.epoch:02d}"
                savedir.mkdir(exist_ok=True)
                
                plots = []

                for images in [self.model.real_A, self.model.fake_B, self.model.cycle_A, self.model.fake_A, self.model.real_B, self.model.cycle_B]:
                    # B X Y Z C
                    images = images[0, :, :, :9].detach().cpu().numpy()  # -> X Y Z C
                    images = images.transpose(2, 1, 0, 3)  # -> Z Y X C
                    images = images.reshape(3, 3, *images.shape[1:])  # -> 3, 3, Y, X, C
                    images = np.concatenate(images, axis=1)  # -> 3, 3Y, X, C
                    images = np.concatenate(images, axis=1)  # -> 3Y, 3X, C
                    plots.append(images)

                for j, title in enumerate(["Source", "Fake B", "Rec A", "Fake A", "Target", "Rec B"]):
                    ax = fig.add_subplot(2, 3, j+1)
                    ax.imshow(plots[j], origin="lower")
                    ax.set_title(title)
                
                plt.savefig(savedir / f"train-{i:06d}.png")

                plt.close()

                self.gen_metrics.reset()
                self.disc_metrics.reset()
                self.log.info(f"Iteration {i:5d}/{len(self.A_dtl_train)}")

            if self.cfg.setting.debug and i > 100:
                break

        self.G_scheduler.step()
        self.D_scheduler.step()

        return self.gen_metrics.compute(), self.disc_metrics.compute()

    @torch.no_grad()
    def test_one_epoch(self):
        for i, ((source_filenames, source_afm, _, _),
                (target_filenames, target_afm, _,
                 _)) in enumerate(zip(self.A_dtl_test, self.B_dtl_test)):
            Z = source_afm.shape[3]
            source_afm = source_afm.to(self.device, non_blocking=True)
            target_afm = target_afm.to(self.device, non_blocking=True)

            fake_source = self.model.to_A(target_afm)  # B X Y Z C
            fake_target = self.model.to_B(source_afm)  # B X Y Z C

            source_afm = (source_afm * 255).to(torch.uint8)
            target_afm = (target_afm * 255).to(torch.uint8)
            fake_source = (fake_source * 255).to(torch.uint8)
            fake_target = (fake_target * 255).to(torch.uint8)

            source_afm = source_afm.permute(0, 3, 4, 1, 2).flatten(0, 1).repeat(1, 3, 1, 1)  # (B Z) C X Y
            target_afm = target_afm.permute(0, 3, 4, 1, 2).flatten(0, 1).repeat(1, 3, 1, 1)
            fake_source = fake_source.permute(0, 3, 4, 1, 2).flatten(0, 1).repeat(1, 3, 1, 1)
            fake_target = fake_target.permute(0, 3, 4, 1, 2).flatten(0, 1).repeat(1, 3, 1, 1)

            self.source_fid.update(fake_source, False)
            self.source_fid.update(source_afm, True)

            self.target_fid.update(fake_target, False)
            self.target_fid.update(target_afm, True)

            if i > 0 and i % self.cfg.setting.log_every == 0:
                self.log.info(f"Test iteration {i:5d}/{len(self.A_dtl_test)}")
                
                savedir = self.outdir / f"Epoch{self.epoch:02d}"
                savedir.mkdir(exist_ok=True)
                
                for b, (source_filename, target_filename) in enumerate(zip(source_filenames, target_filenames)):
                    fig = plt.figure(figsize=(10, 10))
                    fig.suptitle(f"{source_filename} <-> {target_filename}")
                    for j, images in enumerate([source_afm, target_afm, fake_source, fake_target]):
                        images = images[b * Z:b * Z + 9].cpu().numpy()  # 9 C X Y
                        images = images.transpose(0, 3, 2, 1)  # 9 Y X C
                        images = images.reshape(3, 3, *images.shape[1:])  # 3, 3, Y, X, C
                        images = np.concatenate(images, axis=1)  # 3, 3Y, X, C
                        images = np.concatenate(images, axis=1)  # 3Y, 3X, C
                        ax = fig.add_subplot(2, 2, 1 + j)

                        ax.imshow(images, origin="lower")
                        ax.set_title(["Source", "Target", "Fake Source", "Fake Target"][j])
                    
                    plt.savefig(savedir / f"test-{i:06d}.png")
                    plt.close()

            if self.cfg.setting.debug and i > 50:
                break

        return self.source_fid.compute(), self.target_fid.compute()

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
        self.log.info(f"Generator parameters: {sum([p.numel() for p in self.model.G_to_A.parameters()])}")
        self.log.info(f"Discriminator parameters: {sum([p.numel() for p in self.model.D_A.parameters()])}")
        
        self.log.info(f"Source Dataset: {self.cfg.dataset.source_path}")
        self.log.info(f"Target Dataset: {self.cfg.dataset.target_path}")
        
    def save_model(self, metric = None):
        if metric is None or metric < self.best:
            self.best = metric
            if metric is None:
                path = self.outdir / f"cyc_it{self.iters}.pkl"
            else:
                path = self.outdir / f"cyc_it{self.iters}_fid{metric:.4f}.pkl"
            if len(self.save_paths) >= self.cfg.setting.max_save:
                os.remove(self.save_paths.pop(0))
            torch.save(self.model.state_dict(), path)
            self.save_paths.append(path)


def main():
    args = get_parser()

    cfg = Config()

    outdir = Path(args.outdir) / f"{time.strftime('%Y%m%d-%H%M%S')}-cycle"
    
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
