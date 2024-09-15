# NeuGraspNet: Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering

**Authors:** [Snehal Jauhri](https://pearl-lab.com/people/snehal-jauhri), [Ishikaa Lunawat](https://ishikaalunawat.github.io/), and [Georgia Chalvatzaki](https://pearl-lab.com/people/georgia-chalvatzaki)  
**Institution:** [PEARL Lab, TU Darmstadt, Germany](https://pearl-lab.com)  
**Published at:** Robotics: Science and Systems, 2024

**Project Site:** [https://sites.google.com/view/neugraspnet](https://sites.google.com/view/neugraspnet)  
**Paper:** [https://arxiv.org/pdf/2306.07392](https://arxiv.org/pdf/2306.07392)

<p float="left">
  <img src="neugraspnet.gif" width="720"/>
</p>

## Release timeline:
- **11th July 2024:** Initial release with pre-trained weights and simulated grasping demos
- **September 2024:** Dataset generation
- **October 2024:** Training, Inference & ROS package

## Installation

*Tested on Ubuntu 20.04 with an NVIDIA GPU (Recommended 8GB GPU VRAM or higher)*

#### With environment.yml:
- Create a conda environment using the provided environment.yml file:
    ```
    cd <this repo>
    conda env create -f environment.yml
    ```
#### Or with manual conda installation:
- Create a new conda environment with Python 3.8 or higher:
    ```
    conda create --name neugraspnet python=3.8 
    ```
- Install requirements:  
    (Due to compatibility issues with newer versions of open3d, sklearn installation needs to be enabled:)
    ```
    conda activate neugraspnet
    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
    pip install -r requirements.txt
    ```
- Install torch-scatter based on pytorch version and cuda version (https://github.com/rusty1s/pytorch_scatter). For example:
    ```
    pip install torch==1.13.0 torch-scatter==2.1.0 torchvision==0.14.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    ```
- Install the neugraspnet package:
  ```
  cd <this repo>
  pip install -e .
  ```
- Build the conv_occupancy_network dependency:
    ```
    python neugraspnet/scripts/convonet_setup.py build_ext --inplace
    ```

## Run:
- To run evaluations on the pile object dataset from [VGN](https://github.com/ethz-asl/vgn), run:
    ```
    cd neugraspnet/neugraspnet
    python -u scripts/test/sim_grasp_multiple.py --num-view 1 --object_set pile/test --scene pile --num-rounds 100 --model ./data/networks/neugraspnet_pile_efficient.pt --resolution=64 --type neu_grasp_pn_deeper_efficient --qual-th 0.5 --max_grasp_queries_at_once 40 --result-path ./data/results/neu_grasp_pile_efficient --sim-gui
    ```
    Modify the `max_grasp_queries_at_once` command line arguement based on your available GPU memory. (For eg. If using an RTX 3090, use `max_grasp_queries_at_once= 40 or 60`)
- To run evaluations on the egad object dataset (https://dougsm.github.io/egad/), run:
    ```
    python -u scripts/test/sim_grasp_multiple.py --num-view 1 --object_set egad --scene egad --num-rounds 100 --model ./data/networks/neugraspnet_pile_efficient.pt --resolution=64 --type neu_grasp_pn_deeper_efficient --qual-th 0.5 --max_grasp_queries_at_once 40 --result-path ./data/results/neu_grasp_egad_efficient --sim-gui
    ```

### Acknowledgements:
- GPG (https://github.com/atenpas/gpg)
- Convolutional Occupancy Networks (https://github.com/autonomousvision/convolutional_occupancy_networks)
- UNISURF (https://github.com/autonomousvision/unisurf)
- GIGA (https://github.com/UT-Austin-RPL/GIGA)
- Edge-Grasp-Network (https://github.com/HaojHuang/Edge-Grasp-Network)
- VGN (https://github.com/ethz-asl/vgn)
