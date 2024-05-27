# TODO: 
# 1. Copy training loop from example/example.ipynb
# 2. Integrate Hydra for less insane config handling
import os, sys
import wandb
import hydra
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset import get_train_loader
from model import get_model
from pretrainer import get_pretrainer



@hydra.main(
    config_path='../configs', 
    config_name='ae_training.yaml',
    version_base=None
)
def main(
    cfg
):
    # get data
    train_loader, valid_loader = get_train_loader(
        cfg=cfg
    )

    # get model
    model = get_model(
        cfg=cfg, 
    )

    # get trainer
    trainer = get_pretrainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    
    # fit model
    trainer.fit()

if __name__ == "__main__":
    main()

