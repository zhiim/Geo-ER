import csv
import random

from torch.utils.data import IterableDataset, get_worker_info

from functions import compute_dist


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=1000):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        shuffle_buffer = []  # a buffer for shuffle
        # first, fill in this buffer
        try:
            dataset_iter = iter(self.dataset)
            for _ in range(self.buffer_size):
                shuffle_buffer.append(next(dataset_iter))
        except BaseException:
            self.buffer_size = len(shuffle_buffer)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    rand_idx = random.randint(0, self.buffer_size - 1)
                    # fetch one item
                    yield shuffle_buffer[rand_idx]
                    # add another item
                    shuffle_buffer[rand_idx] = item
                except StopIteration:
                    break
            # pop out remained items
            while len(shuffle_buffer) > 0:
                yield shuffle_buffer.pop()
        except GeneratorExit:
            pass


class POIPairDataset(IterableDataset):
    def __init__(self, csv_path):
        super().__init__()

        self.csv_path = csv_path

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            self.len = sum(1 for _ in reader)

    def __len__(self):
        return self.len

    def __iter__(self):
        worker_info = get_worker_info()

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            if worker_info is None:
                for line in reader:
                    yield self._pre_process(line)
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

                for i, line in enumerate(reader):
                    if i % num_workers == worker_id:
                        yield self._pre_process(line)

    def _pre_process(self, line):
        (
            _,
            source_name,
            source_address,
            source_coord,
            pair_name,
            pair_address,
            pair_coord,
            label,
        ) = line

        source_text = "名称 " + source_name + " 地址 " + source_address
        pair_text = "名称 " + pair_name + " 地址 " + pair_address

        lat1, lon1 = [float(item) for item in source_coord.split(",")]
        lat2, lon2 = [float(item) for item in pair_coord.split(",")]
        dist = compute_dist(lat1, lon1, lat2, lon2)

        label = int(label)

        return (source_text, pair_text, dist, label)
