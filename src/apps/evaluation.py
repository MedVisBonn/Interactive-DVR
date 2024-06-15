import os, sys
import hydra
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset import get_eval_dataset
from model import get_model
from utils import get_features, simulate_user_interaction

@hydra.main(
    config_path='../configs', 
    config_name='eval.yaml',
    version_base=None
)
def main(
    cfg
):
    verbose = cfg.verbose

    # get dataset
    dataset = get_eval_dataset(
        cfg=cfg,
        initial_annotation=True,
        verbose=verbose
    )

    # get model and load state dict
    model, state_dict = get_model(
        cfg=cfg,
        return_state_dict=True,
        verbose=verbose
    )
    model.load_state_dict(state_dict)

    # get features
    features = get_features(
        model=model,
        dataset=dataset,
        tta=cfg.tta,
        verbose=verbose
    )

    # get results
    uncertainty_measures = [
        cfg.uncertainty_measure
    ]
    if cfg.background_bias:
        uncertainty_measures.append('feature-distance')

    results = simulate_user_interaction(
        dataset=dataset,
        features=features,
        uncertainty_measures=uncertainty_measures,
        cfg=cfg,
        verbose=verbose
    )




if __name__ == "__main__":
    main()