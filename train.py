import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from dataset import Dataset
from model import Model
from evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')
parser.add_argument('-l', '--logdir', default='./logs', help='directory to write logs')
parser.add_argument('-r', '--restore_checkpoint', default=None,
                    help='path to restore checkpoint, e.g. ./logs/model-100.tar')


def _loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


def _train(path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_log_dir,
           path_to_restore_checkpoint_file):
    batch_size = 128
    initial_patience = 20
    num_steps_to_show_loss = 10
    num_steps_to_check = 100

    step = 0
    patience = initial_patience
    best_accuracy = 0.0
    duration = 0.0

    model = Model()
    model.cuda()
    if path_to_restore_checkpoint_file is not None:
        assert os.path.isfile(path_to_restore_checkpoint_file), '%s not found' % path_to_restore_checkpoint_file
        step = model.load(path_to_restore_checkpoint_file)
        print 'Model restored from file: %s' % path_to_restore_checkpoint_file

    train_loader = torch.utils.data.DataLoader(Dataset(path_to_train_lmdb_dir), batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    evaluator = Evaluator(path_to_val_lmdb_dir)
    optimizer = optim.Adam(model.parameters())

    path_to_losses_npy_file = os.path.join(path_to_log_dir, 'losses.npy')
    if os.path.isfile(path_to_losses_npy_file):
        losses = np.load(path_to_losses_npy_file)
    else:
        losses = np.empty([0], dtype=np.float32)

    while True:
        for batch_idx, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            logits = model.train()(images)
            loss = _loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            duration += time.time() - start_time

            if step % num_steps_to_show_loss == 0:
                examples_per_sec = batch_size * num_steps_to_show_loss / duration
                duration = 0.0
                print '=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                    datetime.now(), step, loss.data[0], examples_per_sec)

            if step % num_steps_to_check != 0:
                continue

            losses = np.append(losses, loss.cpu().data.numpy())
            np.save(path_to_losses_npy_file, losses)

            print '=> Evaluating on validation dataset...'
            accuracy = evaluator.evaluate(model)
            print '==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy)

            if accuracy > best_accuracy:
                path_to_checkpoint_file = model.save(path_to_log_dir, step=step)
                print '=> Model saved to file: %s' % path_to_checkpoint_file
                patience = initial_patience
                best_accuracy = accuracy
            else:
                patience -= 1

            print '=> patience = %d' % patience
            if patience == 0:
                return


def main(args):
    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    path_to_log_dir = args.logdir
    path_to_restore_checkpoint_file = args.restore_checkpoint

    if not os.path.exists(path_to_log_dir):
        os.makedirs(path_to_log_dir)

    print 'Start training'
    _train(path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_log_dir, path_to_restore_checkpoint_file)
    print 'Done'


if __name__ == '__main__':
    main(parser.parse_args())
