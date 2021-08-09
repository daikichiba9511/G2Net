from dataclasses import dataclass
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.metrics import roc_auc_score
from src import utils
from torch.utils.data import DataLoader, Dataset


# ============
# data
# ============
class MyG2NetDataset(Dataset):
    def __init__(self, data_dir: Path, df=None, is_test: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.is_test = is_test
        if is_test:
            self.phase = "test"
            self.id = df["id"].values
            self.data_paths = self.get_data_path(self.phase)
        else:
            self.phase = "train"
            self.id = df["id"].values
            self.target = df["target"].values
            self.data_paths = self.get_data_path(self.phase)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        id_ = self.id[index]
        data = self.get_data(id_)
        if self.is_test:
            return {"data": data}
        target = self.target[index]
        return {"target": target, "data": data}

    def get_data_path(self, phase):
        assert phase in {"train", "test"}
        data_paths = list(self.data_dir.glob(f"{phase}/*/*/*/*.npy"))
        return data_paths

    def load_signal(self, path: Path):
        return np.load(path)

    def make_spectrogram(self, signal):
        spectrogram = np.apply_along_axis(
            partial(librosa.feature.melspectrogram, sr=2048)
            1,
            signal,
        )
        return spectrogram

    def get_data(self, index):
        path = self.data_paths[index]
        signal = self.load_signal(path)
        images = self.make_spectrogram(signal)
        return images


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # define by config
        self.data_dir = config.data_dir
        self.train_df_path = config.train_df_path
        self.fold = config.fold
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size

        # define only
        self.train_df = None
        self.valid_df = None

    def load_train_df(self):
        return pd.read_csv(self.train_df_path)

    def split_train_valid(self, df, fold):
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        train_df = self.load_train_df()
        train_df, valid_df = self.split_train_valid(train_df, fold=self.fold)
        self.train_df = train_df
        self.valid_df = valid_df

    def get_df(self, phase):
        assert phase in {"train", "valid"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df

    def get_dataset(self, phase):
        dataset = MyG2NetDataset(
            data_dir=self.data_dir, df=self.get_df(phase), is_test=(phase == "test")
        )
        return dataset

    def get_batch_size(self, phase):
        assert phase in {"train", "valid"}
        if phase == "train":
            return self.train_batch_size
        elif phase == "valid":
            return self.valid_batch_size

    def get_loader(self, phase):
        # TODO: shuffleとかは考えた方がいいかも
        dataset = self.get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.get_batch_size(phase),
            shuffle=True if phase == "train" else False,
            num_workers=self.num_workers,
            drop_last=True if phase == "train" else False,
        )

    def train_dataloader(self):
        return self.get_loader(phase="train")

    def valid_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


# ============
# model
# ============
class Custom2LinearHead(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, drop_rate=0.5):
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
        self.backbone_name = config.backbone_name
        self.model = timm.create_model(self.backbone_name, pretrained=config.pretrained)
        self.head = Custom2LinearHead(in_features=1000, hidden_size=256, out_features=2)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output


class G2NetLitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # model
        self.net = G2NetModel(config)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.score_fn = roc_auc_score

        # training config
        self.lr = config.lr
        self.T_max = config.T_max
        self.weight_decay = config.weight_decay

        self.save_hyperparameters()

    def calc_acc(self, output, target):
        y_pred = torch.round(torch.sigmoid(output))
        acc = (y_pred == target).sum().float() / len(target)
        return acc

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        target = batch["target"]
        data = batch["data"]
        output = self.forward(data)
        loss = self.loss_fn(target, torch.sigmoid(output))
        acc = self.calc_acc(output, target)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch["target"]
        data = batch["data"]
        output = self.forward(data)
        loss = self.loss_fn(target, output)
        acc = self.calc_acc(output, target)
        score = self.score_fn(
            target.detach().cpu().numpy(), output.detach().cpu().numpy()
        )
        self.log_dict(
            {"valid_loss": loss, "score": score, "valid_acc": acc},
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=self.lr)
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
    base_dir: Path = Path(".")
    output_dir: Path = base_dir / "output"
    debug: bool = True

    # wandb
    prj_name = utils.get_config().prj_name
    entity = utils.get_config().entity

    # data
    data_dir: Path = base_dir / "input" / "g2net-gravitational-wave-detection"
    train_data_path: Path = data_dir / "fold_train_df.csv"

    # model
    backbbone_name: str = "efficientnet_b0"
    pretrained: bool = True

    # strategy
    epochs: int = 30
    fp16: bool = True

    # optim
    lr: float = 5e-3
    weight_decay: float = 5e-6
    T_max: int = epochs


def train(expname, fold, config):
    seed_everything(config.seed, workers=True)
    wandb_logger = WandbLogger(
        name=expname,
        save_dir=str(config.output_dir),
        project=config.prj_name,
        entity=config.entity,
        log_model=False,
    )
    csv_logger = CSVLogger(save_dir=str(config.base_dir / "logs"), name=expname)
    loggers = [wandb_logger, csv_logger]

    checkpoint = ModelCheckpoint(
        dirpath=str(config.output_dir),
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
        amp_backend="native",
        gpus=1,
        benchmark=False,
        default_root_dir=Path.cwd(),
        deterministic=True,
        limit_train_batches=0.1 if config.debug else 1.0,
        limit_val_batches=0.1 if config.debug else 1.0,
        callbacks=[checkpoint],
        logger=loggers,
    )

    datamodule = MyDataModule(config)
    g2netlitmodel = G2NetLitModel(config)
    trainer.fit(datamodule=datamodule, model=g2netlitmodel)


def main():
    config = Config()
    runfile_name = Path(__file__)
    expname = runfile_name.name.split(".")[0] + f"_fold{fold}"
    train(expname, fold=0, config=config)


if __name__ == "__main__":
    main()
