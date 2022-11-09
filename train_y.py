import argparse
import os
from xml.dom.minidom import Entity
import pandas as pd
from tqdm.auto import tqdm
from datetime import timedelta, datetime

from omegaconf import OmegaConf

import wandb
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import re

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# os.chdir("/opt/ml")
wandb_dict = {
    'gwkim_22':'f631be718175f02da4e2f651225fadb8541b3cd9',
    'rion_':'0d57da7f9222522c1a3dbb645a622000e0344d36',
    'daniel0801':'b8c4d92272716adcb1b2df6597cfba448854ff90',
    'seokhee':'c79d118b300d6cff52a644b8ae6ab0933723a59f',
    'dk100':'263b9353ecef00e35bdf063a51a82183544958cc'
}

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=cfg.data.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.lr = config.train.learning_rate

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.MSELoss()  ##MSE LOSS

    def forward(self, x):
        x = self.plm(x)['logits']
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='base_config')
    args, _ = parser.parse_known_args()
    
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    #os.environ["WANDB_API_KEY"] = wandb_dict[cfg.wandb.wandb_username]
    wandb.login(key = wandb_dict[cfg.wandb.wandb_username])
    model_name_ch = re.sub('/','_',cfg.model.model_name)
    wandb_logger = WandbLogger(
                log_model="all",
                name=f'{model_name_ch}_{cfg.train.batch_size}_{cfg.train.learning_rate}_{time_now}',
                project=cfg.wandb.wandb_project,
                entity=cfg.wandb.wandb_entity
                )

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_pearson',
                                        save_top_k=1,
                                        save_last=True,
                                        save_weights_only=False,
                                        verbose=False,
                                        mode='max')

    # Earlystopping
    earlystopping = EarlyStopping(monitor='val_pearson', patience=2, mode='max')
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(cfg.model.model_name, cfg.train.batch_size, cfg.data.shuffle, cfg.path.train_path, cfg.path.dev_path,
                            cfg.path.test_path, cfg.path.predict_path)
    model = Model(cfg)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        gpus=-1, 
        max_epochs=cfg.train.max_epoch, 
        log_every_n_steps=cfg.train.logging_step,
        logger=wandb_logger,    # W&B integration
        callbacks = [checkpoint_callback, earlystopping]
        )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    output_dir_path = 'output'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_path = os.path.join(output_dir_path, f'{model_name_ch}_{time_now}_model.pt')
    torch.save(model, output_path)
