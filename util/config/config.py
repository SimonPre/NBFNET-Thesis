import pathlib

from dacite import from_dict
from omegaconf import OmegaConf, DictConfig

from util.config.conf_dataclass import RunConfig


def read_base_conf() -> DictConfig:
    """
    Reads the base conf yaml
    :return:
    """
    base_file = pathlib.Path(__file__).parent / "base_conf.yaml"
    return OmegaConf.load(base_file)


def creat_config(user_config: dict = None) -> RunConfig:
    """ Creates a config given a user config and the base config"""
    conf = read_base_conf()
    if user_config:
        conf.merge_with(OmegaConf.create(user_config))
    return from_dict(RunConfig, OmegaConf.to_container(conf))
