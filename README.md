# Clear : Contrastive Learning Enhanced Automated Recognition approach for SCVs

This repo is a paper of python implementation : Contrastive Learning Enhanced Automated Recognition approach for SCVs

# Framework

![The overview of GPANet](figs/model.drawio.png)

The overview of our proposed method Clear is illustrated in the Figure, which consists of three modules: .

# Required Packages
- python 3+
- transformers 4.26.1
- pandas 1.5.3
- pytorch 1.13.1

# Datasets
Our proposed method is empirically evaluated on a dataset. Following the methodology of [Qian et al., 2021](https://github.com/Messi-Q/Cross-Modality-Bug-Detection), we conduct experiments to assess reentrancy, timestamp dependence, and integer overflow/underflow vulnerabilities on the dataset.

Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset), which is constantly being updated to provide more details.

# Running
To run program, please use this command: python `run.py`.

Also all the hyper-parameters can be found in `run.py`.

Examples:

`
python run.py --dataset RE --mlmloss 0.1  
`

