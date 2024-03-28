# COMQ

This repository contains the code for paper [COMQ: A Backpropagation-Free Algorithm for Post-Training Quantization](https://arxiv.org/abs/2403.07134). 
The current release includes the following features:

## Files

* `comq.py`: efficient implementations of COMQ
* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantization algorithm
* `quantize_main.py`: code to run quantization

## Dependencies

* `torch`: tested on v1.10.1+cu113
* `timm`: tested on v0.6.12
* `numpy`: tested on v1.19.5
* `tqdm`: tested on v4.62.3
* `scipy`: tested on v1.5.4

All experiments were run on a single 40GB NVIDIA A40. However, some experiments will work on a GPU with a lot less memory as well.

## Usage 

First, make sure ImageNet has been downloaded (we use the ILSVRC-2012 version).

### Run COMQ

```
# Quantize Vision Transformer for 4 bit
python quantize_main.py vit_small_patch16_224 --data_path --batchsize 1024 --wbits 4 --greedy

# Quantize Vision Transformer for 2 bit with batchtuning
python quantize_main.py vit_small_patch16_224 --data_path --batchsize 1024 --wbits 4 --greedy --batchtuning --scalar 0.65
```

# BibTex

```
@misc{zhang2024comq,
      title={COMQ: A Backpropagation-Free Algorithm for Post-Training Quantization}, 
      author={Aozhong Zhang and Zi Yang and Naigang Wang and Yingyong Qin and Jack Xin and Xin Li and Penghang Yin},
      year={2024},
      eprint={2403.07134},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
