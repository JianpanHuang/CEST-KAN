# CEST-KAN
This repository demonstrates an example of CEST MRI data analysis using Kolmogorov-Arnold Networks (KAN) and Lorentzian-KAN (LKAN) for the paper "CEST MRI Data Analysis using Kolmogorov-Arnold Network (KAN) and Lorentzian-KAN (LKAN) Models" submitted to Magnetic Resonance in Medicine. The original KAN can be found [here](https://github.com/KindXiaoming/pykan), while the original efficient KAN can be found [here](https://github.com/Blealtan/efficient-kan).

1. Training can be performed using train_cest_mlp.py, train_cest_kan.py and train_cest_lkan.py for MLP and KAN, respectively. Due to ethical issues, the original training data was not uploaded. Instead, 100,000 Z spectra generated based on real data (randomly selected and with noise added), together with their targets were uploaded for reproduction purposes. The training results are consistent with those shown in the paper (KAN outperforms MLP).

2. The files (py/mat) starting with ‘fig’ are used to generate the corresponding figures in the paper.


Reference:
Wang J#, Cai P#, Wang Z, Zhang H, Huang J*. CEST-KAN: Kolmogorov-Arnold Networks for CEST MRI Data Analysis[J]. arXiv preprint arXiv:2406.16026, 2024.
