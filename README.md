# TimeCIB
This is an official repository of [TimeCIB: Clinical Time Series Imputation using Conditional Information Bottleneck](https://openreview.net/forum?id=wsFCbuDwxY) ([NeurIPS 2023 DGM4H Workshop](https://sites.google.com/ethz.ch/dgm4h-neurips2023))

TimeCIB defines time-series imputation as an information-theoretic perspective. Along with this viewpoint, we found that direct application of Information Bottleneck (IB) framework such as VAE to time series data without considering temporal context can lead to a substantial loss of temporal dependencies, especially on interpolation or extrapolation. To address such a challenge, we propose a novel conditional information bottleneck (CIB) approach for time series imputation, which aims to mitigate the potentially negative consequences of the regularization constraint by reducing the redundant information conditioned on the temporal context.

## Dependency
`python, pytorch`

## Data Preparation
Download data to `./data` using the following script: `bash data/load_{hmnist, physionet}.sh`

Code for rotatedMNIST will be provided soon.

## Reproduce the TimeCIB

- Retrain on the HMNIST: `python train.py --d hmnist --pn hmnist-TimeCIB --rn reproduce --m timecib --b 0.1 --l 0.5 --clen 1.0 --p norm --ep 40 --dim 128 --imputed forward`

- Retrain on the physionet: `python train.py --d physionet --pn physionet-TimeCIB --rn reproduce --m timecib --b 0.001 --l 0.001 --clen 32 --p norm  --ep 50 --dim 16 --imputed forward`

To test your model, use the following.
- Test on the HMNIST: `python train.py --d hmnist --pn hmnist-TimeCIB --rn reproduce --m timecib --b 0.1 --l 0.5 --clen 1.0 --p norm --ep 40 --dim 128 --imputed forward --dir {path_to_the_model} --test True`

- Test on the physionet: `python train.py --d physionet --pn physionet-TimeCIB --rn reproduce --m timecib --b 0.001 --l 0.001 --clen 32 --p norm  --ep 50 --dim 16 --imputed forward --dir {path_to_the_model} --test True`

One can also train/test baseline models (GP-VAE, HI-VAE, VAE). For example, `python train.py --d physionet --pn physionet-Baselines --rn reproduce --m {gpvae, hivae, vae} --b 0.001 --l 0.001 --clen 32 --p {norm, gp}  --ep 50 --dim 16 --imputed forward` Note to use `--p gp` if you want to use GP-VAE.
