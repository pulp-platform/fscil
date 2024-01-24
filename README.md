# 12 mJ per Class On-Device Online Few-Shot Class-Incremental Learning

### Yoga Esa Wibowo, Cristian Cioflan, Thorir Mar Ingolfsson, Michael Hersche, Leo Zhao, Abbas Rahimi, Luca Benini


This repository implements Online Few-Shot Class-Incremental Learning (OFSCIL) in PyTorch and enables the deployment of OFSCIL on the ultra-low power GAP9 microcontroller. OFSCIL is a an extreme edge methodology for enabling pretrained, deployed models to learn new classes using few labeled examples, without forgetting previously learned classes. 

If you use our methodology, please cite the following publication:

```
_DATE'24_
```


## Requirements

The `conda` software is required for running the code. Generate a new environment with

```
$ conda create --name ofscil_env python=3.8
$ conda activate ofscil_env
```

We need PyTorch 1.9.1 and CUDA. Note that the license of each individual package might differ from the license under which the current repository is released.

```
$ (ofscil_env) conda install pytorch=1.9.1=*cuda* torchvision cudatoolkit=11.1 -c pytorch -c nvidia
$ (ofscil_env) pip install -r requirement_all.yml
```
## Datasets

Our codebase includes the necessary code to conduct experiments on miniImageNet and CIFAR100 datasets. The code structure is similar to [C-FSCIL](http://arxiv.org/abs/2203.16588). We follow the [FSCIL](https://github.com/xyutao/fscil) setting and use the same data index_list for training. The CIFAR100 dataset will be downloaded automatically. For miniImageNet, you can download the dataset from [this link](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Once downloaded, please place the file under the `data/` folder and extract it.

```    
$ (ofscil_env) cd data/
$ (ofscil_env) gdown 1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U
$ (ofscil_env) tar -xvf miniimagenet.tar 
```


### Training Base Floating Point Model

To run a the model training including pretraining (session 0), meta learning (session 0), and few-shot class incremental learning accuracy testing (all sessions), use the provided shell script under directory `code`. The script will call `code/main.py` as the executor to do training and testing.

Run main experiments on CIFAR100 with MobileNetV2_x4
```bash
# Training
$ (ofscil_env) cd code

$ (ofscil_env) chmod +x run_mnetv2_cifar100.sh

$ (ofscil_env) ./run_mnetv2_cifar100.sh
```

Run main experiments on CIFAR100 with ResNet12
```bash
# Training
$ (ofscil_env) cd code

$ (ofscil_env) chmod +x run_resnet_cifar100.sh

$ (ofscil_env) ./run_resnet_cifar100.sh
```

Run main experiments on miniImageNet with MobileNetV2_x4
```bash
# Training
$ (ofscil_env) cd code

$ (ofscil_env) chmod +x run_mnetv2_imagenet.sh

$ (ofscil_env) ./run_mnetv2_imagenet.sh
```


Run main experiments on miniImageNet with ResNet12
```bash
# Pretraining
$ (ofscil_env) cd code

$ (ofscil_env) chmod +x run_resnet_imagenet.sh

$ (ofscil_env) ./run_resnet_imagenet.sh
```

In the shell script, you can change experiment id (EXP_ID="X") and the specify which GPU you are using (CUDA_DEVICES="Y"). Multi GPU training is not yet supported. 


### Quantization

The quantisation is performed with quantlab and quantlib library. Initialize and update the submodules, ensuring commit `2ef6dde1a38262935c8498da096c8320a477992e` for [Quantlab](https://github.com/pulp-platform/quantlab) and commit `d76700c2896ce5027ac85e9ca4dbd94d52aa24a3` for `[Quantlib](https://github.com/pulp-platform/quantlib)`.

To do quantization go to `fscil/code` and run `python quantize.py`. Inside the file, you can change (EXPERIMENT_ID="X") to match your desired experiment identifier. This quantization script is only tested on MobileNetV2_x4.

### Deployment

The deployment is conducted using [GAP8 SDK](https://github.com/GreenWaves-Technologies/gap_sdk)/[PULP SDK](https://github.com/pulp-platform/pulp-sdk) and [DORY](https://github.com/pulp-platform/dory). Make sure that both the SDK and DORY are in the same directory with `fscil` directory. When installing and cloning the branch, please check `branch_id.log` to match our working branch. Visit the SDK and DORY repositories for full detail instalation. Copy the files we provide in the current repository under `dory/` directory to the cloned `dory` repository. 

To run the FSCIL code on the microcontroller, you can copy file `script_dory.sh` to DORY folder then execute the program. Edit the EXP_ID to match your exported integerized experiment id. This script will run the system on GVSOC simulator. If you want to upload the program to the chip, you should update the platform name in the script.

```bash
# Pretraining
$ (ofscil_env) cd ..

$ (ofscil_env) cp fscil/script_dory.sh dory

$ (ofscil_env) cd dory

$ (ofscil_env) chmod +x script_dory.sh

$ (ofscil_env) ./script_dory.sh
```

## Acknowledgment

Our code is based on 
- [FSCIL](https://github.com/xyutao/fscil) (Dataset)
- [CEC](https://github.com/icoz69/CEC-CVPR2021) (Dataloader)
- [DeepEMD](https://github.com/icoz69/DeepEMD) (ResNet12)
- [C-FSCIL](http://arxiv.org/abs/2203.16588) (Code structure)


## Contributors

Yoga Esa Wibowo, ETH Zurich, [ywibowo@ethz.ch](ywibowo@ethz.ch)
Cristian Cioflan, ETH Zurich [cioflanc@iis.ee.ethz.ch](cioflanc@iis.ee.ethz.ch)
Thorir Mar Ingolfsson, ETH Zurich [thoriri@iis.ee.ethz.ch](thoriri@iis.ee.ethz.ch)
Michael Hersche, IBM Research Zurich [her@zurich.ibm.com](her@zurich.ibm.com)
Leo Zhao, ETH Zurich [lezhao@student.ethz.ch](lezhao@student.ethz.ch)


## License

Unless explicitly stated otherwise, the code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.
As an exception, the data under the directory `./data` is released under Creative Commons Attribution-NoDerivatives 4.0 International, see `./data/LICENSE` for details..


