import os
import sys
from dataclasses import dataclass
# from functools import partial
from pathlib import Path
from typing import Any, Dict, List

# import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from loguru import logger
from nnAudio.Spectrogram import CQT
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, tensorboard  # , WandbLogger
from sklearn.metrics import roc_auc_score

import math
sys.path.append(str(Path.cwd()))
import matplotlib.pyplot as plt
import seaborn as sns
# import torch_xla.debug.metrics as met
from src import utils
from torch.fft import fft, ifft  # , rfft
from torchaudio.functional import bandpass_biquad
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

plt.style.use("ggplot")


# ============
# data
# ============
def get_transform(phase):
    if phase == "train":
        return A.Compose([
            ToTensorV2(),
        ])
    if phase == "valid":
        return A.Compose([
            ToTensorV2(),
        ])

def get_qtransform(fmin, fmax, hop_length, window):
    return CQT(
        sr=2048,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        # bins_per_octave=8,
        window=window
    )


class MyG2NetDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        df=None,
        is_test: bool = False,
        is_valid: bool = False,
        num_labels: int = 2,
        config=None,
        transform=None
    ):
        super().__init__()
        self.df = df
        self.num_labels = num_labels
        self.data_dir = data_dir
        self.is_test = is_test
        if is_test:
            self.phase = "test"
        else:
            self.phase = "train"
            if is_valid:
                self.phase = "valid"
            # self.target = np.identity(self.num_labels)[df["target"].values]
            self.target = df["target"].values
        self.id = df["id"].values
        self.data_paths = self.get_data_path(self.phase)
        self.dtype = torch.float

        self.transform = transform
        self.wave_qtransform = get_qtransform(
            fmin=config.cqt_fmin,
            fmax=config.cqt_fmax,
            hop_length=config.cqt_hop_length,
            # bins_per_octave=8,
            window=config.cqt_window
        )
        # self.tukey_window = self.get_tukey_window()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id_ = self.id[index]
        data = self.get_data(index)
        data = torch.tensor(data, dtype=self.dtype)
        if self.is_test:
            return {"id": id_, "data": data}
        target = self.target[index].reshape(-1)
        target = torch.tensor(target, dtype=self.dtype)
        return {"id": id_, "target": target, "data": data}

    def get_data_path(self, phase):
        assert phase in {"train", "test", "valid"}
        if phase == "test":
            return list(self.data_dir.glob("test/*/*/*/*.npy"))
        return list(self.data_dir.glob("train/*/*/*/*.npy"))

    def load_signal(self, path: Path):
        signal = np.load(path)
        # normalize: if not use this, almost signal are zero as float32
        signal = signal / np.max(signal, axis=1).reshape(3, 1)
        # CQT needs 1 channel : 3 channels -> 1 channel
        signal = np.hstack(signal)
        return signal

    def get_data(self, index):
        """ important process
        """
        path = self.data_paths[index]
        signal = self.load_signal(path)
        signal = self.whiten(signal)
        signal = torch.from_numpy(signal)
        signal = self.butter_bandpass_filter(signal, lowcut=20, highcut=512)
        image = self.wave_qtransform(signal)
        image = image.squeeze().numpy()
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            image = image.numpy()
        return image

    def whiten(self, signal):
        # window = self.get_torch_tukey(size)
        window = torch.hann_window(len(signal), periodic=True, dtype=self.dtype)
        spec = fft(torch.from_numpy(signal).float() * window)
        mag = torch.sqrt(torch.real(spec * torch.conj(spec)))
        return torch.real(ifft(spec / mag)).numpy() * np.sqrt(len(signal) / 2)

    def get_torch_tukey(self, size, alpha, dtype):
        window = torch.zeros((size, ), dtype=dtype)
        for n in range(size // 2 + 1):
            if n < alpha * size / 2:
                window[n] = window[-n] = 1 / 2 * (1 - math.cos(2 * 3.14 * n / (size * alpha)))
            else:
                window[n] = window[-n] = 1
        return window

    def butter_bandpass_filter(self, data, lowcut, highcut, fs=2048):
        return bandpass_biquad(data, fs, (highcut + lowcut) / 2, (highcut - lowcut) / (highcut + lowcut))


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # define by config
        self.num_workers = config.num_workers
        self.num_labels = config.num_labels
        self.data_dir = config.data_dir
        self.train_df_path = config.train_df_path
        self.test_df_path = config.test_df_path
        self.fold = config.fold
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size

        # define only
        self.train_df = None
        self.valid_df = None

        self.config = config

    def load_train_df(self):
        df = pd.read_csv(self.train_df_path)
        return df

    def load_test_df(self):
        df = pd.read_csv(self.test_df_path)
        return df

    def split_train_valid(self, df, fold):
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        test_df = self.load_test_df()
        train_df = self.load_train_df()
        train_df, valid_df = self.split_train_valid(train_df, fold=self.fold)
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def get_df(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        else:
            return self.test_df

    def get_dataset(self, phase):
        dataset = MyG2NetDataset(
            data_dir=self.data_dir,
            df=self.get_df(phase),
            is_test=(phase == "test"),
            is_valid=(phase == "valid"),
            config=self.config,
            transform=get_transform(phase=phase)
        )
        return dataset

    def get_batch_size(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_batch_size
        elif phase == "valid":
            return self.valid_batch_size
        else:
            return self.test_batch_size

    def get_loader(self, phase):
        dataset = self.get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.get_batch_size(phase),
            shuffle=True if phase == "train" else False,
            num_workers=self.num_workers,
            drop_last=True if phase == "train" else False,
            pin_memory=False,
        )

    def train_dataloader(self):
        return self.get_loader(phase="train")

    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


# ============
# model
# ============
class Custom2LinearHead(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, drop_rate=0.5):
        super(Custom2LinearHead, self).__init__()
        self.in_features = in_features
        self.hiddin_size = hidden_size
        self.out_features = out_features
        self.l0 = torch.nn.Linear(in_features, hidden_size, bias=True)
        self.l1 = torch.nn.Linear(hidden_size, out_features, bias=True)
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.l0(x)
        x = self.dropout(x)
        x = self.l1(x)
        return x


class G2NetModel(nn.Module):
    def __init__(self, config):
        super(G2NetModel, self).__init__()
        self.backbone_name = config.backbone_name
        self.model = timm.create_model(
            self.backbone_name,
            pretrained=config.pretrained,
            in_chans=1,
            # num_classes=0,
            # global_pool="",
        )
        self.model.classifier = Custom2LinearHead(
            in_features=self.model.classifier.in_features,
            hidden_size=256,
            out_features=config.num_labels,
        )
        """
        self.wave_qtransform = get_qtransform(
            fmin=config.cqt_fmin,
            fmax=config.cqt_fmax,
            hop_length=config.cqt_hop_length,
            # bins_per_octave=8,
            window=config.cqt_window
        )
        self.transform = get_transform(phase=self.phase)
        """
    def forward(self, x):
        # x = self.wave_qtransform(x)
        # x = x.unsqueeze(1)
        output = self.model(x)
        return output


class G2NetLitModel(pl.LightningModule):
    def __init__(self, config):
        super(G2NetLitModel, self).__init__()
        # model
        self.net = G2NetModel(config)
        if config.sync_bn:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.score_fn = roc_auc_score

        # training config
        self.lr = config.lr
        self.T_max = config.T_max
        self.weight_decay = config.weight_decay

        # meta
        self.oof_df_path = config.output_dir / config.expname.split("_")[0] / "oof_df.csv"
        self.oof_df = pd.DataFrame() if not self.oof_df_path.exists() else pd.read_csv(self.oof_df_path)
        self.best_score = -1
        self.fold = config.expname.split("_")[1]
        self.exp_name = config.expname
        self.output_dir = config.output_dir / config.expname.split("_")[0]

        self.save_hyperparameters()

    def calc_acc(self, output, target):
        y_pred = torch.argmax(output)
        acc = (y_pred == target).sum().float() / len(target)
        return acc

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        target = batch["target"]
        data = batch["data"]
        output = self(data)
        output = torch.sigmoid(output)
        loss = self.loss_fn(target, output)
        acc = self.calc_acc(output, target)
        if self.trainer.is_global_zero:
            self.log(
                "train_loss", loss, on_epoch=True, prog_bar=True, rank_zero_only=True
            )
            self.log(
                "train_acc", acc, on_epoch=True, prog_bar=True, rank_zero_only=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch["target"]
        data = batch["data"]
        output = self.forward(data)
        output = torch.sigmoid(output)
        loss = self.loss_fn(target, output)
        acc = self.calc_acc(output, target)

        target = target.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        score = self.score_fn(target, output)
        if self.trainer.is_global_zero:
            self.log(
                "valid_loss", loss, on_epoch=True, prog_bar=True, rank_zero_only=True
            )
            self.log("score", score, on_epoch=True, prog_bar=True, rank_zero_only=True)
            self.log(
                "valid_acc", acc, on_epoch=True, prog_bar=True, rank_zero_only=True
            )
        return {"target": target, "logits": output, "score": score}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        score = np.array([out["score"] for out in outputs])
        max_score = np.max(score)
        if max_score > self.best_score:
            self.best_score = max_score
            # target = np.concatenate([out["target"] for out in outputs])
            logits = np.concatenate([out["logits"] for out in outputs])
            # self.oof_df[f"fold{self.fold}_target"] = target
            # self.oof_df[f"fold{self.fold}_logits"] = logits
            # self.oof_df.to_csv(self.oof_df_path, index=False)

            plt.subplots(figsize=(10, 8))
            # sns.displot(target, label="target", kde=True)
            sns.distplot(logits, label="logits", kde=True)
            plt.title("oof plot")
            plt.legend()
            plt.savefig(f"./{self.output_dir}/oof_plot_{self.fold}.png")
            plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.net.parameters(), weight_decay=self.weight_decay, lr=self.lr
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.T_max
        )
        return [optimizer], [lr_scheduler]


# ============
# config
# ============
@dataclass
class Config:
    # meta
    seed = 42
    num_workers = os.cpu_count()
    base_dir: Path = Path(".")
    output_dir: Path = base_dir / "output"
    debug: bool = False
    fold: int = 0
    expname: str = ""

    # wandb
    prj_name = utils.get_config().prj_name
    entity = utils.get_config().entity

    # data
    data_dir: Path = Path("./") / "input" / "g2net-gravitational-wave-detection"
    df_data_dir: Path = Path("./") / "input" / "g2net-gravitational-wave-detection"
    train_df_path: Path = df_data_dir / "fold_train_df.csv"
    test_df_path: Path = df_data_dir / "sample_submission.csv"

    # model
    backbone_name: str = "efficientnet_b7"
    pretrained: bool = True

    # preprocess
    # type of windows , see here : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
    cqt_window = "hann"
    cqt_fmin: int = 20
    cqt_fmax: int = 1024
    cqt_hop_length: int = 32

    # strategy
    max_grad_norm: float = 100
    sync_bn: bool = False
    num_labels: int = 1
    epochs: int = 20
    train_batch_size: int = 8 * 4
    valid_batch_size: int = 8 * 4
    fp16: bool = True
    device = "cuda" if torch.cuda.is_available() else "tpu"
    tpu_cores: int = 1

    # optim
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-6
    T_max: int = epochs


def train(expname, fold, config):
    # wandb_logger = WandbLogger(
    # name=expname,
    # save_dir=str(config.output_dir),
    # project=config.prj_name,
    # entity=config.entity,
    # log_model=False,
    # )
    csv_logger = CSVLogger(save_dir=str(config.base_dir / "logs"), name=expname)
    tensor_logger = tensorboard.TensorBoardLogger(
        save_dir=str(config.base_dir / "tb_logs"), name=expname
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(config.output_dir / expname.split("_")[0]),
        filename=f"{expname}_fold{fold}" + "{epoch:02d}",
        save_weights_only=True,
        save_top_k=1,
        monitor="score",
        mode="max",
    )

    trainer = Trainer(
        max_epochs=config.epochs if not config.debug else 1,
        precision=16 if config.fp16 else 32,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0 if not config.debug else 2,
        amp_backend="native",
        gpus=1 if config.device == "cuda" else 0,
        tpu_cores=config.tpu_cores if config.device == "tpu" else None,
        gradient_clip_val=config.max_grad_norm,
        benchmark=False,
        default_root_dir=Path.cwd(),
        deterministic=True,
        limit_train_batches=0.1 if config.debug else 1.0,
        limit_val_batches=0.1 if config.debug else 1.0,
        callbacks=[checkpoint],
        logger=[
            # wandb_logger,
            csv_logger,
            tensor_logger,
        ],
        progress_bar_refresh_rate=20,
        # plugins=pl.plugins.training_type.TPUSpawnPlugin(debug=True),
    )

    datamodule = MyDataModule(config)
    g2netlitmodel = G2NetLitModel(config)
    # print(met.metrics_report())
    trainer.fit(datamodule=datamodule, model=g2netlitmodel)


def make_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=0)
    parser.add_argument("--tpu_cores", default=1)
    parser.add_argument("--debug", default=False)
    args = parser.parse_args()
    return args


def main():
    runfile_name = Path(__file__)

    # make settings from args
    args = make_args()
    fold = args.fold
    tpu_cores = args.tpu_cores
    debug = args.debug
    expname = runfile_name.name.split(".")[0] + f"_fold{fold}"

    # wandb.login(key=utils.get_config().wandb_token)

    config = Config(expname=expname, fold=fold, tpu_cores=tpu_cores, debug=debug)
    (config.output_dir / expname.split("_")[0]).mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, workers=True)
    # print(met.metrics_report())
    train(expname, fold=config.fold, config=config)


if __name__ == "__main__":
    main()