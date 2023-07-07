# Following rules apply to setting up NeuralRecon and ZED2 camera on a NVIDIA Jetson Orin. 

## 1. First boot setup: 

`https://www.stereolabs.com/blog/getting-started-with-jetson-agx-orin-devkit/`

Install JetPack components.  

```
sudo apt update 
sudo apt dist-upgrade 
sudo reboot 
sudo apt install nvidia-jetpack 
```
Next, download and setup the latest ZED SDK for Jetson from Stereolabs' website. Run the downloaded package (Substituting `<l4t_version>` and `<ZED_SDK_version>` with the version you downloaded). When prompted, no need to install python API as we will set up a conda environment later. 

```
cd ~/Downloads # replace with the correct folder if required 
chmod +x ZED_SDK* 
./ZED_SDK_Tegra_L4T<l4t_version>_v<ZED_SDK_version>.run  
```

## 2. Setting up NeuralRecon dependencies 

`https://github.com/zju3dv/NeuralRecon`

Clone and enter the NeuralRecon repository. Make sure conda is installed. 

Modify `environment.yaml` in the directory to as follows: 
 
```
name: neucon 
channels: 
    - pytorch 
    - defaults 
    - conda-forge 
dependencies: 
    - python=3.8.10
    - pytorch==1.12.1
    - torchvision==0.13.1 
    - cudatoolkit=11.3 
    - ipython 
    - tqdm 
    - numba 
    - sparsehash # dependency for torchsparse 
    - pip 
    - pip: 
        - -r requirements.txt 
        - git+https://github.com/mit-han-lab/torchsparse.git 
```

After that, activate the "neucon" environment. 
 
```
sudo apt install libsparsehash-dev
conda env create -f environment.yaml 
conda activate neucon 
```

## 3. Setting up the ZED python SDK 

`https://www.stereolabs.com/docs/app-development/python/install/`

Install prerequisites: 

```
python3 -m pip install cython numpy opencv-python pyopengl
```

Next, `cd` into your local ZED SDK folder and call the script to install the ZED python SDK. This may take a few tries since the script has trouble fetching the URL sometimes. 

```
cd "/usr/local/zed/" 
python3 get_python_api.py 
```

## 4. Setting up the Orin specific Pytorch

`https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html`

In order to interface with the Jetson's CUDA correctly, you need to re-install a specific pytorch version from NVIDIA. Begin with the requirements:

```
sudo apt-get -y update; 
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
```

Next, install pytorch.

```
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --force-reinstall --no-cache $TORCH_INSTALL
```

Add these following environment variables to the end of your bash configuration file `~/.bashrc`.

Reload your terminal.

```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} 
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libffi.so.7 /usr/lib/aarch64-linux-gnu/libgomp.so.1" 
export LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH 
export CPATH=$CPATH:/usr/local/cuda/include 
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64 
export CUDA_HOME=/usr/local/cuda 
```

And you're done! Both NeuralRecon and ZED 2 examples should run. For NeuralRecon, additional datasets/checkpoints/models may require setup; refer to their readme for more details.

## 5. Running the ZED demo

The ZED demo `zed_demo.py` is based on the ARkit demo example provided in the NeuralRecon repository, with the added functionality of data collection using a connected ZED camera.

To run the example, copy the NeuralRecon folder on to the NeuralRecon repository (No files should be overwritten). Ensure the `PATH` variable in the config file is pointing to where you want the data to be saved. Then,
```
python zed_demo.py --cfg ./config/zed_demo.yaml
```
