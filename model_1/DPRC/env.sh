#!/bin/bash
set -e

source "${HOME}/conda/etc/profile.d/conda.sh"
conda create -n holo python=3.11 --yes
conda activate holo

conda install nvidia/label/cuda-12.1.1::cuda-cudart-dev nvidia/label/cuda-12.1.1::cuda-toolkit nvidia/label/cuda-12.1.1::libcublas-dev nvidia/label/cuda-12.1.1::libcufft-dev nvidia/label/cuda-12.1.1::libcurand-dev nvidia/label/cuda-12.1.1::libcusolver-dev nvidia/label/cuda-12.1.1::libcusparse-dev --yes
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate wandb ipykernel gpustat peft sentencepiece bitsandbytes ipywidgets trl accelerate huggingface_hub matplotlib opencv-python deepspeed fire tyro ipdb nvitop seaborn tldr
pip install flash-attn opencv-python-headless python-dotenv rich debuglater[all] pre-commit
pip install absl-py autograd imageio mpmath numpy Pillow plotly pytorch-msssim torch-dct
pip install scipy scikit-learn scikit-image
pip install tensorboardx tensorflow tensorflow-estimator tensorflow-probability
pip install tqdm ml-collections medpy SimpleITK h5py timm einops nvitop rich python-box
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
pushd "$SCRIPT_DIR/external_libraries/compressai"
pip install -e .
popd
