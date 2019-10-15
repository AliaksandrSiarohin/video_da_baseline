from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader

from datasets import VideoFeaturesDataset
from loss import *
from models import BaselineModel


def main(args):
    source_dataset = VideoFeaturesDataset(args.features_folder_source, list_file=args.list_file_source,
                                          num_frames=args.num_frames, sampling_strategy='TSNTrain')

    source_dataset = DataLoader(source_dataset, args.bs, shuffle=True, drop_last=True)

    target_dataset = VideoFeaturesDataset(args.features_folder_target, list_file=args.list_file_target,
                                          num_frames=args.num_frames, sampling_strategy='TSNTrain')

    target_dataset = DataLoader(target_dataset, args.bs, shuffle=True, drop_last=True)

    assert len(source_dataset) == len(target_dataset)  # Increase min dataset size

    val_target_dataset = VideoFeaturesDataset(args.features_folder_target, list_file=args.list_file_val,
                                              num_frames=args.num_frames, sampling_strategy='TSNVal',
                                              min_dataset_size=0)

    val_target_dataset = DataLoader(val_target_dataset, args.bs, shuffle=False, drop_last=False)

    model = BaselineModel(dial_last=args.dial_last, num_classes=source_dataset.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    for epoch in range(args.num_epochs):
        print('Starting epoch %d / %d ' % (epoch + 1, args.num_epochs))

        run_epoch(model, source_dataset, target_dataset, optimizer, args)

        source_acc = check_accuracy(model, source_dataset)
        target_acc = check_accuracy(model, target_dataset)
        val_acc = check_accuracy(model, val_target_dataset)

        print('Source acc: %f , Train acc: %f, Train Val acc: %f' % (source_acc, target_acc, val_acc))


def run_epoch(model, source_dataset, target_dataset, optimizer, args):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for source, target in zip(source_dataset, target_dataset):
        x_source = source['frames'].cuda()
        y_source = source['labels'].cuda().long()

        x_target = target['frames'].cuda()

        # Run a forward pass and compute the score and loss
        score_source = model(x_source)
        score_target = model(x_target)

        loss_source = criterion(score_source, y_source)
        loss_target = args.entropy_target_weight * entropy(score_target)
        loss = (loss_source + loss_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(model, loader):
    model.eval()
    num_correct, num_samples = 0, 0

    with torch.no_grad():
        for x in loader:
            x = x['frames'].cuda()
            y = x['labels'].cuda().long()

            scores = model(x)

            _, preds = torch.max(scores, 1)
            num_correct += torch.sum(preds == y.data)
            num_samples += x.size(0)

        acc = float(num_correct) / num_samples

    return acc


if __name__ == "main":
    parser = ArgumentParser()

    parser.add_argument('--list_file_source', type=str, default="data/hmdb51/list_hmdb51_train_hmdb_ucf-feature.txt")
    parser.add_argument('--list_file_target', type=str, default="data/ucf101/list_ucf101_train_hmdb_ucf-feature.txt")

    parser.add_argument('--list_file_val', type=str, default="data/ucf101/list_ucf101_val_hmdb_ucf-feature.txt")

    parser.add_argument('--features_folder_source', default='/media/gin/data/dataset/hmdb51')
    parser.add_argument('--features_folder_target', default='/media/gin/data/dataset/ucf101')
    parser.add_argument('--num_frames', type=int, default=5)

    parser.add_argument('--entropy_target_weight', type=float, default=3)

    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--dial_last', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
