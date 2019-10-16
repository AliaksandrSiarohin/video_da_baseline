from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from datasets import VideoFeaturesDataset, SeqRandomSampler
from loss import entropy
from models import BaselineModel


def main(args):
    source_dataset = VideoFeaturesDataset(args.features_folder_source, list_file=args.list_file_source,
                                          num_frames=args.num_frames, sampling_strategy='TSNTrain')
    num_classes = source_dataset.num_classes

    target_dataset = VideoFeaturesDataset(args.features_folder_target, list_file=args.list_file_target,
                                          num_frames=args.num_frames, sampling_strategy='TSNTrain')

    num_samples = max(len(source_dataset), len(target_dataset))

    source_dataset = DataLoader(source_dataset, args.bs, shuffle=False, num_workers=args.num_workers, drop_last=True,
                                sampler=SeqRandomSampler(source_dataset, num_samples=num_samples))
    target_dataset = DataLoader(target_dataset, args.bs, shuffle=False, num_workers=args.num_workers, drop_last=True,
                                sampler=SeqRandomSampler(target_dataset, num_samples=num_samples))

    val_target_dataset = VideoFeaturesDataset(args.features_folder_target, list_file=args.list_file_val,
                                              num_frames=args.num_frames, sampling_strategy='TSNVal')

    val_target_dataset = DataLoader(val_target_dataset, args.bs, shuffle=False, drop_last=False)

    model = BaselineModel(dial_last=args.dial_last, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    #schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20])

    model = model.cuda()

    for epoch in range(args.num_epochs):
        print('Starting epoch %d / %d ' % (epoch + 1, args.num_epochs))

        run_epoch(model, source_dataset, target_dataset, optimizer, args)

        source_acc = check_accuracy(model, source_dataset)
        target_acc = check_accuracy(model, target_dataset)
        val_acc = check_accuracy(model, val_target_dataset)

        #schedule.step(epoch)

        print('Source acc: %f , Target acc: %f, Train Val acc: %f' % (source_acc, target_acc, val_acc))


def run_epoch(model, source_dataset, target_dataset, optimizer, args):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for source, target in zip(source_dataset, target_dataset):
        x_source = source['frames'].cuda()
        y_source = source['labels'].cuda().long()

        x_target = target['frames'].cuda()

        # Run a forward pass and compute the score and loss
        score_source = model(x_source, True)
        score_target = model(x_target, False)

        loss_source = criterion(score_source, y_source)
        loss_target = args.entropy_target_weight * entropy(score_target)
        loss = (loss_source + loss_target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def check_accuracy(model, loader):
    model.eval()
    num_correct, num_samples = 0, 0

    with torch.no_grad():
        for x in loader:
            y = x['labels'].cuda().long()
            x = x['frames'].cuda()

            scores = model(x, False)

            _, preds = torch.max(scores, 1)
            num_correct += torch.sum(preds == y.data)
            num_samples += x.size(0)

        acc = float(num_correct) / num_samples

    return acc


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--source', type=str, default='ucf101')
    parser.add_argument('--target', type=str, default='olympic')

    parser.add_argument('--list_file_source', type=str,
                        default="data/{source}/list_{source}_train_{target}-feature.txt")
    parser.add_argument('--list_file_target', type=str,
                        default="data/{target}/list_{target}_train_{source}-feature.txt")

    parser.add_argument('--list_file_val', type=str, default="data/{target}/list_{target}_val_{source}-feature.txt")

    parser.add_argument('--features_folder_source', default='/media/gin/data/dataset/{source}/RGB-feature')
    parser.add_argument('--features_folder_target', default='/media/gin/data/dataset/{target}/RGB-feature')
    parser.add_argument('--num_frames', type=int, default=5)

    parser.add_argument('--entropy_target_weight', type=float, default=0.3)
    parser.add_argument('--dial_last', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    args.list_file_source = args.list_file_source.format(**vars(args))
    args.list_file_target = args.list_file_target.format(**vars(args))
    args.list_file_val = args.list_file_val.format(**vars(args))
    args.features_folder_source = args.features_folder_source.format(**vars(args))
    args.features_folder_target = args.features_folder_target.format(**vars(args))

    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
