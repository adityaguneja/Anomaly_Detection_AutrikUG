from anomalib.data import Folder
#from anomalib.models import Padim #for training on systems with higher VRAM
#from anomalib.models import Patchcore   
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint
import torch
import gc


gc.collect()
torch.cuda.empty_cache()                    #makes sure maximum amount of VRAM is available for learning
torch.set_float32_matmul_precision('medium')#decareses VRAM usage


if __name__ == "__main__":                  #required for anomalib scripts to work

    
    datamodule = Folder(
        name="my_dataset",                  #name of folder where dataset exists
        root="data/train",      #directory where training images exist in folder
        normal_dir="good",                  #Folder with normal images
        abnormal_dir="anomaly",             #Folder with anomalous images          
        train_batch_size=1,                 #**must be 1 for EfficientAD**              #Adjust batch size as needed - higher:faster training,more VRAM
        eval_batch_size=8,                  #Batch size for eval/test
        num_workers=8,                      #Tune for CPU Usage
        
    )
    datamodule.setup()

    model = EfficientAd(
        
        imagenet_dir="./datasets/imagenette", #EfficientAD Specific existingdataset
        teacher_out_channels=384,
        model_size= 'medium',                 #medium improves model performance, slower  
        lr=5e-5,                              #learning rate also impacts model performance
        weight_decay=1e-5,
        padding=False,
        pad_maps=True
    )

    '''model = Padim()'''
    '''model = Patchcore(
        backbone="wide_resnet50_2",            #To use other available models
        coreset_sampling_ratio=0.1
    )'''
      
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",         #Directory to save checkpoints, necessary if splitting datasets
        filename="best",               #Checkpoint File name pattern (no extension needed)
        #filename="best",
        monitor="train_loss",          # Metric to monitor (use a metric relevant to your task)--helpful when using early stopping
        mode="min"                     # "min" for loss, "max" for metrics like AUROC/F1
    )

    engine = Engine(                   
        max_epochs=10,                  #adjust according to feasibility
        precision="16-mixed",           #reduces VRAM usage **ONLY Available for EfficienTAD
        callbacks=[checkpoint_callback]
    )

    # Train the model
    engine.fit(datamodule=datamodule, model=model)

   
