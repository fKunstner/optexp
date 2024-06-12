# Installing dependencies

The library is not yet packaged with depencencies on pipy.  

Follow this document to install the dependencies.

Once the depencies are installed, clone the repo and install with `pip install -e .` from the directory containing `setup.cfg`.

## Pytorch

Start by installing Pytorch. We use version `2.3.0` along with the following packages.
- ROCM 6.0 (Linux only)
  ```
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
  pip install torchtext==0.18.0 
  ```
- CUDA 11.8
  ```
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
  pip install torchtext==0.18.0 
  ```
- CUDA 12.1
  ```
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
  pip install torchtext==0.18.0 
  ```
- CPU only
  ```
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
  pip install torchtext==0.18.0 
  ```
- Slurm cluster with pre-compiled wheels
  ```
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 torchtext==1.8.0 --no-index
  ```

## Other dependencies

- To run the library 
  `pip install -r main.txt`
- Dev tools    
  `pip install -r dev.txt`
- Building docs 
  `pip install -r docs.txt`
