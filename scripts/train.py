import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

# 在 runtime 里读 cfg.xxx 时，才会触发插值 resolver 的计算（若尚未解析）
register_default_resolvers()

# cfg 是 Hydra 根据 configs/train.yaml（及 defaults、命令行覆盖、resolver）拼出来的 OmegaConf 对象
# 装饰器在调用 main 函数时将其作为参数注入
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")  # Hydra 的根配置就是 ./configs/train.yaml
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
