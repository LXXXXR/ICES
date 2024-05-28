# Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning

This is the implementation of our paper "[Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning](https://arxiv.org/abs/xxxx)" in ICML 2024. This repo is based on the open-source [pymarl](https://github.com/oxwhirl/pymarl) framework, with some implementation adapted from [CDS](https://github.com/lich14/CDS). Please refer to those repo for more documentation.

## Installation instructions

Set up Google Research Football: Follow the instructions in [GRF](https://github.com/google-research/football?tab=readme-ov-file#quick-start) .


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

For example, run ICES on 3v1:

```
python src/main.py --exp-config=3v1_ices_qplex_s0 --config=ICES_QPLEX --env-config=academy_3_vs_1_with_keeper
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