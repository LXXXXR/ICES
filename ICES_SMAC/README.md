# Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning

This is the implementation of our paper "[Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning](https://arxiv.org/abs/xxxx)" in ICML 2024. This repo is based on the open-source [pymarl2](https://github.com/hijkzzz/pymarl2) framework, and please refer to that repo for more documentation.

## Installation instructions

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

Install Python environment with conda:

```bash
conda create -n ices python=3.8 -y
conda activate ices
```

then install with `requirements.txt` using pip:

```bash
pip install -r requirements.txt
```

## Run an experiment 

```shell
python src/main.py --config=[Algorithm name] --env-config=[Env name] --exp-config=[Experiment name]
```

The config files are all located in `src/config`.

`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.
`--exp-config` refers to the config files in `src/config/exp`. If you want to change the configuration of a particular experiment, you can do so by modifying the yaml file here.

All results will be stored in the `work_dirs` folder.

For example, run ICES on 2s3z:

```
python src/main.py --exp-config=2s3z_ices_s0 --config=ices --env-config=sc2
```

## Citing

If you use this code in your research or find it helpful, please consider citing our paper:
```
@article{li2024individual,
  title={Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning},
  author={Li, Xinran and Liu, Zifan and Chen, Shibo and Zhang, Jun},
  booktitle={accepted by International Conference on Machine Learning (ICML)},
  year={2024}
}
```
