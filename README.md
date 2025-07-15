# Anomaly_Detection_AutrikUG
This project utilises the EfficientAD model of Anomalib framework to detect anomalies such as bridge cracks with high accuracy. It also contains instructions to setup the model on your own system and train the model locally with other models such as Padim and PatchCore. 

Pre-Checks:
1. If running on a Windows system, make sure to run the IDE with Administrator permissions.
2. In case your system does not allow activating virtual environments, use 
  Set-ExecutionPolicy Unrestricted -Scope Process
to disable script check only for the current process and enable script activation.

Setup:
1. Create a virtual environment with Python version 3.10 *latest Python versions are not compatible with Anomalib.
2. Complete Anomalib installation with full dependencies and other core packages using:
    pip install anomalib
    anomalib install --option full**
3. Usage of CUDA cores is recommended if you have an NVIDIA GPU. In case the gpu_torch script fails to identify any available GPU, make sure the PyTorch version installed is CUDA compatible, reinstall the required version using:
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118*
4. Prepare your Dataset by adding images to the train directory of the data folder. Add both cracked and uncracked images. Anomalib uses a very specific data structure; a reference directory has been added for ease.
5. Initiate training of the model by running the train_anomalib script; this will save the model checkpoint to the checkpoints folder. If the dataset is too large and consumes more than available VRAM, use the subsequent train2 scripts to continue training the model by picking up the progress from the checkpoint saved earlier.
6. When training is complete, add test images to the test folder of the data directory.
7. Initiate testing by running the test_modelName script according to the model used for training. Accuracy scores are shown post testing, and results, along with heatmaps and prediction maps, are saved to the result folder.  
