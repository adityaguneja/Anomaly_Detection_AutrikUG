#Use if larger dataset is split

from anomalib.data import Folder 
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium') 


if __name__ == "__main__":


    checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",        
            filename="best",              
            monitor="train_loss",         
            mode="min"                    
    )

    datamodule = Folder(
        name="my_dataset",                  
        root="data/my_dataset/train2",      
        normal_dir="good",                  
        abnormal_dir="anomaly",                       
        train_batch_size=1,                
        eval_batch_size=1,                  
        num_workers=4                       
    )
    datamodule.setup()

    #Use the checkpoint saved earlier to continue traiining.
    model = EfficientAd.load_from_checkpoint("checkpoints/best.ckpt")
    
    engine = Engine(
        max_epochs=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback]
    )

    engine.fit(datamodule=datamodule, model=model)

