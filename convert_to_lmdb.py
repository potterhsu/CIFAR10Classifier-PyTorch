import argparse
import os
import cPickle
import random
import lmdb
import example_pb2
from meta import Meta

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to CIFAR-10 batches folder and write the converted files')


class ExampleReader(object):
    def __init__(self, path_to_batch_file):
        with open(path_to_batch_file, 'rb') as f:
            data_batch = cPickle.load(f)
        self._images = data_batch['data']
        self._labels = data_batch['labels']
        self._num_examples = self._images.shape[0]
        self._example_pointer = 0

    def read_and_convert(self):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        image = self._images[self._example_pointer].reshape([32, 32, 3], order='F').transpose(1, 0, 2).tostring()
        label = self._labels[self._example_pointer]
        self._example_pointer += 1

        example = example_pb2.Example()
        example.image = image
        example.label = label
        return example


def convert_to_lmdb(path_to_batch_files, path_to_lmdb_dirs, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_lmdb_dir in path_to_lmdb_dirs:
        num_examples.append(0)
        writers.append(lmdb.open(path_to_lmdb_dir, map_size=10*1024*1024*1024))

    for path_to_batch_file in path_to_batch_files:
        example_reader = ExampleReader(path_to_batch_file)
        finished = False
        while not finished:
            txns = [writer.begin(write=True) for writer in writers]

            for _ in xrange(10000):
                idx = choose_writer_callback(path_to_lmdb_dirs)
                txn = txns[idx]

                example = example_reader.read_and_convert()
                if example is None:
                    finished = True
                    break

                str_id = '{:08}'.format(num_examples[idx] + 1)
                txn.put(str_id, example.SerializeToString())
                num_examples[idx] += 1

            [txn.commit() for txn in txns]

    for writer in writers:
        writer.close()

    return num_examples


def create_lmdb_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_batches_meta_file,
                          path_to_lmdb_meta_file):
    print 'Saving meta file to %s...' % path_to_lmdb_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    with open(path_to_batches_meta_file, 'rb') as f:
        content = cPickle.load(f)
        meta.categories = content['label_names']
    meta.save(path_to_lmdb_meta_file)


def main(args):
    path_to_batches_meta_file = os.path.join(args.data_dir, 'cifar-10-batches-py/batches.meta')
    path_to_batches = os.path.join(args.data_dir, 'cifar-10-batches-py')
    path_to_train_batch_files = [os.path.join(path_to_batches, 'data_batch_1'),
                                 os.path.join(path_to_batches, 'data_batch_2'),
                                 os.path.join(path_to_batches, 'data_batch_3'),
                                 os.path.join(path_to_batches, 'data_batch_4'),
                                 os.path.join(path_to_batches, 'data_batch_5')]
    path_to_test_batch_files = [os.path.join(path_to_batches, 'test_batch')]

    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    path_to_test_lmdb_dir = os.path.join(args.data_dir, 'test.lmdb')
    path_to_lmdb_meta_file = os.path.join(args.data_dir, 'lmdb_meta.json')

    for path_to_dir in [path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_test_lmdb_dir]:
        assert not os.path.exists(path_to_dir), 'LMDB directory %s already exists' % path_to_dir

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_lmdb(path_to_train_batch_files,
                                                             [path_to_train_lmdb_dir, path_to_val_lmdb_dir],
                                                             lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_lmdb(path_to_test_batch_files,
                                          [path_to_test_lmdb_dir],
                                          lambda paths: 0)

    create_lmdb_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_batches_meta_file,
                          path_to_lmdb_meta_file)

    print 'Done'


if __name__ == '__main__':
    main(parser.parse_args())
