Documentation
=============
## Introduction ##
Welcome to the Tumor Growth Simulation ToolKit's documentation page!

## Installation ##
### C++ ###
#### Step 1: Getting the Source Files ####
TGSTK's source files can be retrieved from the [github repo](https://github.com/cormarte/tgstk.git) using:

	git clone https://github.com/cormarte/tgstk.git
	
#### Step 2: Compiling ####
*[Section under writing]*

### Python ###
#### Step 1: Install the CUDA® Toolkit ####
NVIDIA's CUDA® Toolkit v9.2 first needs to be installed from [here](https://developer.nvidia.com/cuda-92-download-archive). After installation, the CUDA® binaries directory needs to be added to the system PATH using:

	SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\bin;%PATH%

Note: The CUDA® binaries directory may need to be adjusted depending on the CUDA® Toolkit installation location.

#### Step 2: Install TGSTK from PyPI ####
A package containg the TGSTK binaries and Python wrappers can be installed directly from PyPI using:

	pip install tgstk


## Getting Started ##
### Tumour Growth Simulation ###
A minimal example file for tumour growth simulation using TGSTK in Python is available [here](https://github.com/cormarte/tgstk/blob/main/src/example.py). To run the example, you will need to: 

- Download [this file](https://github.com/cormarte/tgstk/blob/main/src/vtk_numpy_utils.py) and add it to your Python project directory
- Download and unzip the sample data from [here](https://lisaserver.ulb.ac.be/owncloud/index.php/s/nK9v5u8k3cmFdkr)
- Install Matplotlib in your Python environment, if not already done, using:


	pip install matplotlib

Once done, simply run:

	<project/directory>>python example.py <sample/data/directory>
	
You should get something like:


\image html "../resources/result.png" width=1400px

