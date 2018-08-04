## Machine Learning with Tensorflow

### Step by Step setup environment with Cloud9

##### Install MiniConda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

##### Create virtual environment
conda create -n py3 python=3 ipython
source activate py3
conda install pip

##### Packages
pip install numpy
pip install pandas
pip install matplotlib

##### Tensorflow
conda install -c conda-forge tensorflow


##### Create requirements.txt (venv)
pip freeze --local > requirements.txt



##### For Anaconda Tensorflow-GPU(local):
conda create -n tensorflow pip python=3.5 
activate tensorflow
pip install --ignore-installed --upgrade tensorflow-gpu 

###### CUDA Toolkit (stable)
https://developer.nvidia.com/
(do not install Visual Studio Integration)

https://developer.nvidia.com/cudnn
https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html



### Saving ANN progress
Done

### Improving ANN performance
Done


### Debugging tensorflow

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/debug_demo.ipynb

### Training ANN in Cloud

##### Estimator API

### Setting up Google Cloud

