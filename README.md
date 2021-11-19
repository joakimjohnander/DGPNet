# DGPNet - Dense Gaussian Processes for Few-Shot Segmentation
Welcome to the public repository for DGPNet. The paper is available at arxiv: https://arxiv.org/abs/2110.03674 .

## How to run
### Download data
1. Download and unzip PASCAL and COCO images
2. Download and unzip PASCAL and COCO annotations (we provide link [here](https://liuonline-my.sharepoint.com/:f:/g/personal/joajo88_ad_liu_se/EmKWrLV-q85Cu4nORmKCErQBCVDI4j_M3SFc7zBsC_QBFA?e=KozVaa))
3. Change `local_config.py` to point out the images and annotations. Also change `slurm_launch.sh` if using slurm.
4. Download and unzip PASCAL and COCO data splits (we provide link [here](https://liuonline-my.sharepoint.com/:f:/g/personal/joajo88_ad_liu_se/EmKWrLV-q85Cu4nORmKCErQBCVDI4j_M3SFc7zBsC_QBFA?e=KozVaa))
5. Make sure that the data splits are at `DGPNet/data_splits`

### Install dependencies
The dependencies are listed in `DGPNet/singularity/Dockerfile21.09`

### Train and test model
We typically run via slurm, using
```
sbatch singularity/slurm_launch.sh runfiles/dgp_5shot_pascal_resnet50.py --train --test --dataset pascal --fold 0 --add_packages_to_path
```

## Code layout
- `checkpoints` - Checkpoints will be stored here at the end of training.
- `data_splits` - Defines the different folds.
- `fss` - Code is here.
- `local_config.py` - Used to set up paths
- `logs` - Used to store slurm checkpoints
- `runfiles` - Any experiment we run is defined in a runfile. The runfile is launched as main to start the experiment.
- `singularity` - We use singularity/slurm and any files related to that are stored here.
- `visualization` - During training and testing, our code stores some visualizations. They go here.
