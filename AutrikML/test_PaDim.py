import torch
from anomalib.data import Folder
from anomalib.models import Padim
from sklearn.metrics import roc_auc_score

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

    model = Padim.load_from_checkpoint("checkpoints/best-v1.ckpt")
    model.eval()
    model.cuda()  

    all_scores = []
    all_labels = []


    for batch in datamodule.test_dataloader():
        images = batch.image.cuda()
        labels = batch.gt_label.cpu().view(-1)

        with torch.no_grad():
            outputs = model(images)
            scores = outputs[0].cpu().view(-1)  # For PaDiM, outputs[0] is the anomaly score

        all_scores.append(scores)
        all_labels.append(labels)

    # Concatenate all batches
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Compute AUROC using scikit-learn
    auroc_score = roc_auc_score(all_labels.numpy(), all_scores.numpy())
    print("Test AUROC (scikit-learn):", auroc_score)



    
