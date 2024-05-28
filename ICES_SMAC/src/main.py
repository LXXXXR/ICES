import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
import sys
import torch as th
from utils.logging import get_logger
import yaml
import setproctitle
from pathlib import Path
import time
import os.path as osp
import wandb

from run import REGISTRY as run_REGISTRY


def main(config_dict):
    config = config_copy(config_dict)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    config["env_args"]["seed"] = config["seed"]
    exp_name = config["exp_name"]
    run_name = config["run_name"]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    work_dir = osp.join("work_dirs", exp_name, run_name, timestamp)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    config["work_dir"] = work_dir

    setproctitle.setproctitle(
        "-".join([config["env"], config["name"], config["run_name"]])
    )

    console_logger = get_logger(
        f"{exp_name} - {run_name}",
        log_file=osp.join(work_dir, f"{timestamp}.log"),
    )
    wandb.init(
        project=exp_name,
        name=run_name,
        entity="lxxxxr",
        config=config,
    )
    # run
    if "use_per" in config_dict and config_dict["use_per"]:
        run_REGISTRY["per_run"](config, console_logger)
        raise NotImplementedError("Not implemented yet.")
    else:
        assert config_dict["run"] == "default", "Not Implemented yet."
        run_REGISTRY[config_dict["run"]](config, console_logger)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index("=") + 1 :].strip()
            break
    return result


if __name__ == "__main__":
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    exp_config = _get_config(params, "--exp-config", "exp")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, exp_config)

    main(config_dict)
