import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

# 用 Hydra 加载 configs/train.yaml，然后调用 fastwam.runtime.run_training(cfg)（主训练逻辑在 src/fastwam/runtime.py）
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
