from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import torch
import gc




gc.collect()
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium') 

if __name__ == "__main__":


    datamodule = Folder(
        name="my_dataset",
        root="data/test",#directory where test images exist
        normal_dir="good",          #floder with good(non-cracked) images.
        abnormal_dir="anomaly",     #floder with cracked images.
        train_batch_size=1,         #Must be 1 for EfficientAD.
        eval_batch_size=8,          #Can be higher for evaluation.
        num_workers=8,              #tune for CPU usage   
        
    )
    datamodule.setup(stage="test")  #Explicitly setup the test set

    #Load the trained model from checkpoint, use filename as set during training
    model = EfficientAd.load_from_checkpoint("checkpoints/best.ckpt")


    engine = Engine()
    test_results = engine.test(model=model, datamodule=datamodule)
