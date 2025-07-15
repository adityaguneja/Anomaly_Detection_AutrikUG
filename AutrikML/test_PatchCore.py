from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
import torch
import gc


gc.collect()
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium') 

if __name__ == "__main__":
    
    datamodule = Folder(
        name="my_dataset",
        root="data/test",
        normal_dir="good",
        abnormal_dir="anomaly",
        train_batch_size=1,    
        eval_batch_size=1,     
        num_workers=4,    
    )
    datamodule.setup(stage="test") 

    # Load the trained model from checkpoint
    model = Patchcore.load_from_checkpoint("checkpoints/best.ckpt")

    engine = Engine()
    test_results = engine.test(model=model, datamodule=datamodule)
