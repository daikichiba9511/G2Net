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
from nnAudio.Spectrogram import CQT
# import wandb
# from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, tensorboard  # , WandbLogger
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path.cwd()))
import matplotlib.pyplot as plt
import seaborn as sns
# import torch_xla.debug.metrics as met
from src import utils
from torch.utils.data import DataLoader, Dataset


# ============
# data
# ============
class MyG2NetDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        df=None,
        is_test: bool = False,
        is_valid: bool = False,
        num_labels: int = 2,
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
            self.target = np.identity(self.num_labels)[df["target"].values]
        self.id = df["id"].values
        self.data_paths = self.get_data_path(self.phase)
        self.dtype = torch.float

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id_ = self.id[index]
        data = self.get_data(index)
        data = torch.tensor(data, dtype=self.dtype)
        if self.is_test:
            return {"id": id_, "data": data}
        target = self.target[index]
        target = torch.tensor(target, dtype=self.dtype)
        return {"id": id_, "target": target, "data": data}

    def get_data_path(self, phase):
        assert phase in {"train", "test", "valid"}
        if phase == "test":
            return list(self.data_dir.glob("test/*/*/*/*.npy"))
        return list(self.data_dir.glob("train/*/*/*/*.npy"))

    def load_signal(self, path: Path):
        signal = np.load(path)
        signal = signal / np.max(signal, axis=1).reshape(3, 1)
        signal = np.hstack(signal)
        return signal

    def get_data(self, index):
        path = self.data_paths[index]
        signal = self.load_signal(path)
        return signal


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
            pin_memory=True,
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
        self.wave_transform = CQT(
            sr=2048,
            fmin=20,
            fmax=512,
            hop_length=64,
            # bins_per_octave=8,
        )

    def forward(self, x):
        x = self.wave_transform(x)
        x = x.unsqueeze(1)
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
        output = self.forward(data)
        output = F.softmax(output, dim=1)
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
        output = F.softmax(output, dim=1)
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
        return {"target": target, "logits": output}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        target = np.concatenate([out["target"] for out in outputs])
        logits = np.concatenate([out["logits"] for out in outputs])
        plt.subplots(figsize=(10, 8))
        sns.distplot(target, label="target")
        sns.distplot(logits, label="logits")
        plt.title("oof plot")
        plt.legend()
        plt.savefig(f"./{self.output_dir}/oof_plot_{self.fold}_{self.current_epoch}.png")
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
    print("num_wokers: num_workers")
    base_dir: Path = Path(".")
    output_dir: Path = base_dir / "output"
    debug: bool = False
    fold: int = 0
    expname: str = ""

    # wandb
    prj_name = utils.get_config().prj_name
    entity = utils.get_config().entity

    # data
    data_dir: Path = Path("/content") / "input" / "g2net-gravitational-wave-detection"
    df_data_dir: Path = Path("./") / "input" / "g2net-gravitational-wave-detection"
    train_df_path: Path = df_data_dir / "fold_train_df.csv"
    test_df_path: Path = df_data_dir / "sample_submission.csv"

    # model
    backbone_name: str = "efficientnet_b7"
    pretrained: bool = True

    # strategy
    sync_bn: bool = False
    num_labels: int = 2
    epochs: int = 20
    train_batch_size: int = 8 * 16
    valid_batch_size: int = 8 * 16
    fp16: bool = False

    # optim
    lr: float = 5e-3
    weight_decay: float = 5e-6
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
        # amp_backend="native",
        # gpus=1,
        tpu_cores=1,
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


def main():
    runfile_name = Path(__file__)
    fold = 0
    expname = runfile_name.name.split(".")[0] + f"_fold{fold}"
    # wandb.login(key=utils.get_config().wandb_token)

    config = Config(expname=expname, fold=fold)
    (config.output_dir / expname.split("_")[0]).mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, workers=True)
    # print(met.metrics_report())
    train(expname, fold=config.fold, config=config)


if __name__ == "__main__":
    main()
