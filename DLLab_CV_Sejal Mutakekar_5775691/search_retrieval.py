import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import paths
from datasets.voc12 import VOCDataset, create_image_only_transforms
from models.blip.blip_config import BlipConfig
from models.blip.blip_retrieval import BlipRetrieval
from models.preprocessing.preprocess import get_processors
from utils.logger import setup_logger

def get_top10(eval_ckpt=None, query = "a picture of a plane"):
    # setup model for validation
    setup_logger(level=logging.INFO)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Running on device: {device}, cuda available: {torch.cuda.is_available()}")

    model_cfg = BlipConfig()
    model = BlipRetrieval.from_config(model_cfg)
    #logging.info(f"Created model {type(model).__name__} with {model.show_n_params()} parameters.")
    model.load_checkpoint(Path(paths.CV_PATH_CKPT) / "blip_model_base.pth")
    # todo optionally overwrite this with your own checkpoint
    if eval_ckpt is None:
        eval_ckpt = Path(paths.CV_PATH_CKPT) / "blip_model_retrieval_head.pth"
    model.load_retrieval_head(eval_ckpt)
    model = model.to(device)
    _ = model.eval()

    # setup dataset
    voc_path = Path(paths.CV_PATH_VOC)
    vis_processor_val, text_processor_val = get_processors(model_cfg, mode="eval")
    dataset = VOCDataset(voc_path, voc_path / "ImageSets" / "Segmentation" / "val.txt",
                         load_captions=True, transforms=create_image_only_transforms(vis_processor_val))

    # collect image features for the dataset
    val_dataset = VOCDataset(
        voc_path, voc_path / "ImageSets" / "Segmentation" / "val.txt", load_captions=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            num_workers=0, drop_last=False)
    logging.info(f"Collect image features")

    image_feats, text_feats = [], []
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            logging.info(f"{i}/{len(dataloader)}")
        image = batch["image"].to(device)
        with torch.no_grad():
            image_feat = model.forward_image(image)
            image_feats.append(image_feat.detach().cpu().numpy())
    image_feats = np.concatenate(image_feats, axis=0)


    # define search query

    print(f"Search query: {query}")

    # get the text feature
    with torch.no_grad():
        text_feat = model.forward_text([text_processor_val(query)]).cpu().numpy()

    # compute similarity
    sim = (text_feat @ image_feats.T)[0]

    # show the top10 results
    top10 = {
        "id": np.argsort(-sim)[:10],
        "fname" : [],
        "name" : [],
        "caption" : [],
        "sim" : [],
    }
    for rank, i in enumerate(top10["id"]):
        top10["sim"].append(sim[i])
        top10["fname"].append(dataset.files[i]["img"])
        data = dataset[i]
        top10["caption"].append(data['caption'])
        top10["name"].append(data['name'])

    return top10
