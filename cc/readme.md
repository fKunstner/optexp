# ComputeCanada

Compute Canada specific configuration and installation scripts. 

# Installation process 

Load the main environment configuration
```
source cc/load_modules.sh
``` 

If not already done, create a virtual environment
```
virtualenv --no-download ~/env/optexp
source ~/env/optexp/bin/activate
```

Clone the repository 
```
git clone https://github.com/fkunstner/optexp ~/optexp
cd ~/optexp
```

Install the dependencies
```
pip install -r requirements/main.txt --no-index
pip install -r requirements/torch_121.txt --no-index
pip install -e . --no-index
```

Create a script to load the environment
```
cp envs/env.sh.example envs/cc.sh
# edit envs/cc.sh with your editor of choice
source envs/cc.sh 
```

# Loading 

Some of the above steps need to be repeated on every login on CC.
To avoid the pain, add the following to your `.bashrc` or `.bash_profile`:
```
source ~/optexp/cc/load_modules.sh
source ~/env/optexp/bin/activate
source ~/optexp/envs/cc.sh
```
