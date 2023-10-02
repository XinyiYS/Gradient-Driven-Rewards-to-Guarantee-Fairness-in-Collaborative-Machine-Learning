# Gradient Driven Rewards to Guarantee Fairness in Collaborative Machine Learning [NeurIPS'2021]
Official code repository for our accepted work "Gradient Driven Rewards to Guarantee Fairness in Collaborative Machine Learning" in the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021:

> Xinyi Xu*, Lingjuan Lyu\*, Xingjun Ma, Chenglin Miao, Chuan Sheng Foo, Bryan Kian Hsiang Low
>
> Gradient Driven Rewards to Guarantee Fairness in Collaborative Machine Learning [Paper](https://proceedings.neurips.cc/paper/2021/hash/8682cc30db9c025ecd3fee433f8ab54c-Abstract.html)

### Set up environment using conda

Tested OS platform: Ubuntu 20.04 with Nvidia driver Version: 470.86 CUDA Version: 11.4

` conda env create -f environment.yml`

### Running the `main.py`

Running on _MNIST_ dataset with 5 agents and uniform data split (i.e., I.I.D). Automatically uses GPU if available.

`python main.py -D mnist -N 5 -split uni `

### Results directory

The results are saved in csv formats in a `RESULTS` directory (created if not exist) by default.


## Citing
If you have found our work to be useful in your research, please consider citing it with the following bibtex:
```
@inproceedings{Xu2021,
   author = {Xu, Xinyi and Lyu, Lingjuan and Ma, Xingjun and Miao, Chenglin and Foo, Chuan Sheng and Low, Bryan Kian Hsiang},
   booktitle = {Advances in Neural Information Processing Systems},
   editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
   pages = {16104--16117},
   publisher = {Curran Associates, Inc.},
   title = {Gradient Driven Rewards to Guarantee Fairness in Collaborative Machine Learning},
   volume = {34},
   year = {2021}
}
```
