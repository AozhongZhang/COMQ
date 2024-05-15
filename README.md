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

## Prepare ImageNet Dataset

First, make sure ImageNet has been downloaded (we use the ILSVRC-2012 version). Place the .tar files for training set and validation set both under the data/ILSVRC2012 or your-path/ILSVRC2012.

Then unzip Imagenet dataset:

```
# prepare the training data and move images to subfolders:
mkdir ILSVRC2012_img_train
mv ILSVRC2012_img_train.tar ILSVRC2012_img_train 
cd ILSVRC2012_img_train 
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

# Prepare the validation data:
cd ..
mkdir ILSVRC2012_img_val
mv ILSVRC2012_img_val.tar ILSVRC2012_img_val && cd ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
```

### Run COMQ

```
# Quantize Vision Transformer for 4 bit
python quantize_main.py vit_small_patch16_224 --data_path --batchsize 1024 --wbits 4 --greedy

# Quantize Vision Transformer for 2 bit with batchtuning
python quantize_main.py vit_small_patch16_224 --data_path --batchsize 1024 --wbits 2 --greedy --batchtuning --scalar 0.65
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
