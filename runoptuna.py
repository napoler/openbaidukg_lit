import torch
from transformers import BertTokenizer,AdamW
# from torch.utils.data import 
# from performer_pytorch import PerformerLM
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
# from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split,TensorDataset
import pytorch_lightning as pl
# from tkit_mlp_pytorch import MLP
from transformers import BertTokenizer, BertForMaskedLM,AlbertTokenizer,BertConfig
# import torch\

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# help(DataLoader)
from pytorch_lightning.loggers import WandbLogger
import wandb
# wandb_logger = WandbLogger(project="百度数据做开放关系BD_Knowledge Extractionnotebookb9f0237b84")
# /kaggle/input/reformerchinesemodel/epoch4step21209.ckpt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
# 自动停止
# https://pytorch-lightning.readthedocs.io/en/1.2.1/common/early_stopping.html
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.optim as optim

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

# help(DataLoader)

# /kaggle/input/reformerchinesemodel/epoch4step21209.ckpt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
# 自动停止
# https://pytorch-lightning.readthedocs.io/en/1.2.1/common/early_stopping.html
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

# 引入修剪技术　https://pytorch-lightning.readthedocs.io/en/stable/advanced/pruning_quantization.html
from pytorch_lightning.callbacks import ModelPruning
import torch.nn.utils.prune as prune
# https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html
# 量化　降低内存　低精度　　https://pytorch-lightning.readthedocs.io/en/stable/advanced/pruning_quantization.html
from pytorch_lightning.callbacks import QuantizationAwareTraining

# 使用 DDP 时设置 find_unused_pa​​rameters=False
# 默认情况下，我们已启用查找未使用的参数为 True。这是针对过去出现的兼容性问题（有关更多信息，请参阅讨论）。默认情况下，这会影响性能，并且在大多数情况下可以禁用。
from pytorch_lightning.plugins import DDPPlugin

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
import wandb


# 解决不显示图表问题
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)








class LitAutoLM(pl.LightningModule):
    """
    继承自bertlm模型
    做预测
    """
    def __init__(self, learning_rate=3e-4,T_max=500,optimizer_name="AdamW", **kwargs):
        super().__init__()
        self.hparams.config = BertConfig.from_pretrained("uer/roberta-small-word-chinese-cluecorpussmall")
        
        self.save_hyperparameters()
#         self.tokenizer = AlbertTokenizer.from_pretrained("uer/roberta-base-word-chinese-cluecorpussmall")
#         self.model = BertForMaskedLM.from_pretrained("uer/roberta-base-word-chinese-cluecorpussmall")
# https://huggingface.co/uer/roberta-tiny-word-chinese-cluecorpussmall
#         configuration = BertConfig.from_pretrained("uer/roberta-tiny-word-chinese-cluecorpussmall")

        self.tokenizer = AlbertTokenizer.from_pretrained("uer/roberta-tiny-word-chinese-cluecorpussmall")
        self.model = BertForMaskedLM.from_pretrained("uer/roberta-small-word-chinese-cluecorpussmall",config=self.hparams.config)

#         self.loss_fn = torch.nn.MSELoss()
#         tokenizer.save_vocabulary("./")
    def forward(self, input_ids,token_type_ids=None,attention_mask=None,labels=None):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits
        return outputs
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, token_type_ids,attention_mask,y = batch
        outputs = self(x, token_type_ids,attention_mask,y)
        self.log('train_loss', outputs.loss)
        return outputs.loss
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, token_type_ids,attention_mask,y = batch
        outputs = self(x, token_type_ids,attention_mask,y)
        self.log('val_loss', outputs.loss)
        self.log("hp_metric",outputs.loss) # 这个参数用于参数效果对比
        return outputs.loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, token_type_ids,attention_mask,y = batch
        outputs = self(x, token_type_ids,attention_mask,y)
        self.log('test_loss', outputs.loss)
        
        
        logits = outputs.logits
        for one,one2 in zip(logits.argmax(dim=-1).tolist(),y.tolist()):
            kgtext=[]
            kgtext2=[]
            for it,it2 in zip(one,one2):
    #             print(tokenizer.decode(it))
#                 if tokenizer.decode(it) in ["[SEP]","[PAD]"]:
#                     continue
                kgtext.append(tokenizer.decode(it))
                kgtext.append(tokenizer.decode(it2))
            print(kgtext)
            
            
        return outputs.loss
    def pred(self, text,ner1,ner2):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = self.tokenizer([ner1+"和"+ner2+"关系为"+"".join(["[MASK]"]*10)],[text], return_tensors="pt", padding="max_length",max_length=128,truncation=True)
        outputs = self(**inputs)
        logits = outputs.logits
        kgtext=[]
        full=[]
        words=tokenizer.tokenize(text)
        for it in logits.argmax(dim=-1).tolist()[0]:
#             print(tokenizer.decode(it))
            full.append(tokenizer.decode(it))
            if tokenizer.decode(it) in ["[SEP]","[PAD]"]:
                continue
            
            kgtext.append(tokenizer.decode(it))
#         i="".join(kgtext).index('关系为')
#         return kgtext[i+3],kgtext
        return kgtext,full


    def pred2(self, text,ner1,p):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = self.tokenizer([ner1+p+"为"+"".join(["[MASK]"]*10)],[text], return_tensors="pt", padding="max_length",max_length=128,truncation=True)
        outputs = self(**inputs)
        logits = outputs.logits
        kgtext=[]
        full=[]
        words=tokenizer.tokenize(text)
        for it in logits.argmax(dim=-1).tolist()[0]:
#             print(tokenizer.decode(it))
            full.append(tokenizer.decode(it))
            if tokenizer.decode(it) in ["[SEP]","[PAD]"]:
                continue
            
            kgtext.append(tokenizer.decode(it))
#         i="".join(kgtext).index('关系为')
#         return kgtext[i+3],kgtext
        return kgtext,full
#         x, token_type_ids,attention_mask = batch
#         outputs = self(x, token_type_ids,attention_mask)
#         self.log('test_loss', outputs.loss)
#         return outputs.loss
#     help(model.predict)
    def configure_optimizers(self):
            """优化器 # 类似于余弦，但其周期是变化的，初始周期为T_0,而后周期会✖️T_mult。每个周期学习率由大变小； https://www.notion.so/62e72678923f4e8aa04b73dc3eefaf71"""
    #         optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))

            #只优化部分
#             optimizer = torch.optim.AdamW(self.parameters(), lr=(self.hparams.learning_rate))
            if self.hparams.optimizer_name=="AdamW":
                optimizer = AdamW(self.parameters(), lr=(self.hparams.learning_rate))
            else:
                optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adadelta
#             optimizer = self.get_optimizer()
            #         使用自适应调整模型
            T_mult=2
            scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=self.hparams.T_max,T_mult=T_mult,eta_min=0 ,verbose=False)
    #         https://github.com/PyTorchLightning/pytorch-lightning/blob/6dc1078822c33fa4710618dc2f03945123edecec/pytorch_lightning/core/lightning.py#L1119

            lr_scheduler={
    #            'optimizer': optimizer,
               'scheduler': scheduler,
#                 'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
                'interval': 'step', #epoch/step
                'frequency': 1,
                'name':"lr_scheduler",
                'monitor': 'train_loss', #监听数据变化
                'strict': True,
            }
    #         return [optimizer], [lr_scheduler]
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}









import logging,sys
def objective(trial: optuna.trial.Trial) -> float:
    # wandb_logger = WandbLogger(name='nerchaijie拆解',project="colab_optuna 使用pytorch_lightning超参优化 .ipynb")
    # wandb.init()
    # name=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # name=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(num)
    # num=num+1
    # wandb_logger = WandbLogger(project="colab_optuna 使用pytorch_lightning超参优化 .ipynb")
    # wandb_logger = WandbLogger()
    # We optimize the number of layers, hidden units in each layer and dropouts.
#     n_layers = trial.suggest_int("n_layers", 1, 3)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     max_epochs = trial.suggest_int("max_epochs", 1, 1)
    batch_size = trial.suggest_categorical("batch_size", [12])
# "MomentumSGD",
#     optimizer_name = trial.suggest_categorical("optimizer", ["Adam","AdamW", "RMSprop", "SGD"])
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
    accumulate_grad_batches=trial.suggest_int("accumulate_grad_batches", 1, 5)
#     batch_size=72
    traindataset=torch.load("traindataset.pkt")
    devdataset=torch.load("devdataset.pkt")
    train_loader=DataLoader(traindataset,batch_size=batch_size, shuffle=True, )
    val_loader=DataLoader(devdataset,batch_size=batch_size, shuffle=False, )
    # test_loader=DataLoader(testdataset,batch_size=batch_size, shuffle=False, )
    model=LitAutoLM(learning_rate=learning_rate,optimizer_name=optimizer_name)
    
    trainer = pl.Trainer(
            gpus=1,
        #     min_epochs=1,
            precision=16,amp_level='O2',
#             checkpoint_callback=checkpoint_callback,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    #         resume_from_checkpoint="/kaggle/input/openkgmodelml/wandb/run-20210610_054617-3cu3oiy2/files/百度数据做开放关系BD_Knowledge\ Extractionnotebookb9f0237b84/3cu3oiy2/checkpoints/chinese-out.ckpt",
            auto_select_gpus=True,
#             callbacks=[lr_monitor,early_stop_callback],
            deterministic=True,
    #         auto_scale_batch_size='binsearch',
    #         auto_lr_find=True,
    #         max_epochs=wandb.config.epochs,
            max_epochs=1,
    #         logger=wandb_logger,
    #         accumulate_grad_batches=wandb.config.accumulate_grad_batches

            accumulate_grad_batches=2)

    
    hyperparameters = dict(learning_rate=learning_rate,batch_size=batch_size,optimizer_name=optimizer_name)
    print("hyperparameters",hyperparameters)
    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, train_loader,val_loader)
    # still doesn't work
    val_loss=trainer.callback_metrics["val_loss"].item()
    del model
#     del trainer
    torch.cuda.empty_cache()
    return val_loss



study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, timeout=600)



print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))





