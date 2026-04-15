import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

# 用 Hydra 加载 configs/train.yaml，然后调用 fastwam.runtime.run_training(cfg)（主训练逻辑在 src/fastwam/runtime.py）
# run_training 里用的是 instantiate(cfg.model, ...)，最终调用哪个工厂由 cfg.model._target_ 决定
# 例如 task=libero_uncond_2cam224_1e-4 里 override /model: fastwam，对应 create_fastwam
# 工厂函数路线：来自 configs/model/*.yaml，由 task 的 defaults 里 override /model: ... 选中
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
