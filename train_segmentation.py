import os
from pathlib import Path
import paths
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from utils.logger import setup_logger
import copy
from PIL import Image

from datasets import VOCDataset
from datasets import transforms as tr
from models.segmentation import vit_small, TransformerSegmentationHead, EncoderDecoder
from utils import mean_iou, SegmentationDisplay

torch.manual_seed(0)
np.random.seed(0)


def display_results(ious, classes):
    print(f"\n{'Class':>20}  IoU\n")
    [print(f"{cl:>20}: {iou*100:.02f}") for cl, iou in zip(classes, ious)]
    print(f"{'---':>20}----\n{'Average':>20}: {np.nanmean(ious)*100:.02f}\n")


def main(args):
    setup_logger(level=logging.INFO)

    # setup augmentation
    normalize = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    training_pipeline = tr.Compose([tr.RandomResize(256, 300), tr.RandomCrop(256), tr.RandomHorizontalFlip(0.5), tr.ToTensor(), normalize])
    test_pipeline = tr.Compose([tr.ToTensor(), normalize])

    # setup datasets
    training_dataset = VOCDataset(paths.CV_PATH_VOC, os.path.join(paths.CV_PATH_VOC, "ImageSets/Segmentation/train.txt"), transforms=training_pipeline)
    val_dataset = VOCDataset(paths.CV_PATH_VOC, os.path.join(paths.CV_PATH_VOC, "ImageSets/Segmentation/val.txt"), transforms=test_pipeline)

    # setup loaders
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=args.batch_size, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)

    logging.info(f"Training samples: {len(training_dataset)}, validation samples: {len(val_dataset)}")

    # setup model
    encoder = vit_small()
    ckpt_path = Path(paths.CV_PATH_CKPT) / "dino_deitsmall16_pretrain.pth"
    encoder.load_state_dict(torch.load(ckpt_path))
    logging.info(f"Loaded encoder weights from {ckpt_path}")

    if args.transformer_head_shared_qk and args.decoder_type != "transformer":
        raise ValueError("Shared Q and K valid for transformer decoder only.")

    if args.decoder_type == "linear":
        # implemented as a convolutional layer with kernel size 1
        decoder = nn.Conv2d(in_channels=encoder.embed_dim, out_channels=len(training_dataset.classes), kernel_size=1)
    elif args.decoder_type == "convolutional":
        # START TODO #################
        # implement a small convolutional decoder
        decoder = nn.Conv2d(encoder.embed_dim, len(training_dataset.classes), kernel_size=3, padding=1)
        # END TODO ###################
    elif args.decoder_type == "transformer":
        # START TODO #################
        # complete the implementation of TransformerSegmentationHead and use it as a decoder. Use the same embed_dim and num_heads as the encoder, don't forget the flag for the shared QK attention.
        decoder = TransformerSegmentationHead(len(training_dataset.classes), encoder.embed_dim, encoder.num_heads, shared_qk=args.transformer_head_shared_qk)
        # END TODO ###################
    else:
        raise ValueError

    # concatenate encoder and head/decoder
    model = EncoderDecoder(encoder, decoder).cuda()
    model.eval()

    # optimization items
    loss_function = nn.CrossEntropyLoss(ignore_index=training_dataset.ignore_label)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=1.0)

    # training loop
    for epoch in range(args.epochs):
        logging.info(f"Training epoch {epoch}")
        train_epoch(model, training_loader, loss_function, optimizer)
        ious = validate(model, val_loader)
        display_results(ious, val_dataset.classes)

        logging.info(f"  max GPU memory allocated: {torch.cuda.max_memory_allocated()/1e6:.03f}M")
        scheduler.step()


def train_epoch(model, data_loader, loss_function, optimizer):
    print_loss_interval = 50
    losses = []
    model.train()
    for idx, sample in enumerate(data_loader):
        img = sample['image']
        segm = sample['label']

        # forward pass
        logits = model(img.cuda())
        loss = loss_function(logits, segm.cuda())

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if idx%print_loss_interval==(print_loss_interval-1):
            logging.info(f"Training {idx} - avg. loss: {np.mean(losses):.03f}")
            losses = []


def validate(model, data_loader):
    model.eval()
    seg_viz = SegmentationDisplay(data_loader.dataset.classes, 0.65)

    preds = []
    gts = []
    for idx, sample in enumerate(data_loader):
        img = sample['image']
        segm = sample['label']
        with torch.no_grad():
            logits = model(img.cuda()).cpu()
        preds.append(logits.argmax(1))
        gts.append(copy.deepcopy(segm))

        # example code for saving predictions
        # if idx==0:
        #     seg_viz.draw_and_save(Image.open(sample["img_path"][0]), logits.argmax(1), "output.png")

    iou = mean_iou(preds, gts, len(data_loader.dataset.classes), data_loader.dataset.ignore_label)
    
    return iou


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("decoder_type", type=str, choices=["linear", "convolutional", "transformer"])
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", "-bs", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--transformer-head-shared-qk", action="store_true")
    args = parser.parse_args()
    main(args)



