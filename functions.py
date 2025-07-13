import json
from difflib import SequenceMatcher
from io import open
from math import atan2, cos, radians, sin, sqrt

from transformers import BertTokenizer

import config


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def compute_dist(lat1, lon1, lat2, lon2):
    R = 6373.0

    try:
        float(lat1)
    except ValueError:
        return -1

    try:
        float(lon1)
    except ValueError:
        return -1

    try:
        float(lat2)
    except ValueError:
        return -1

    try:
        float(lon2)
    except ValueError:
        return -1

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))

    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = round(R * c * 1000)
    dist = 2 * (dist) / config.MAX_DIST - 1

    return dist


def get_lat_long(entity):
    words = entity.lower().split()
    for i, word in enumerate(words):
        if words[i - 2] == "latitude" and words[i - 1] == "val":
            latitude = float(word)
            longitude = float(words[i + 4])
            idx = i

        if words[i - 2] == "postalcode" and words[i - 1] == "val":
            try:
                words[i] = str(int(float(words[i])))
            except ValueError:
                pass

    del words[idx - 3 : idx + 5]

    return " ".join(words), latitude, longitude


def prepare_dataset(path, n_path, max_seq_len=128):
    tokenizer = BertTokenizer.from_pretrained("./bert")

    data_x = []
    coord_x = []
    data_y = []
    neigh_x = []

    with open(path, "r") as f:
        for line in f:
            arr = line.split("\t")

            e1, lat1, long1 = get_lat_long(arr[0])
            e2, lat2, long2 = get_lat_long(arr[1])

            if len(arr) > 2:
                x = tokenizer.tokenize(
                    "[CLS] " + e1 + " [SEP] " + e2 + " [SEP]"
                )
                y = arr[2]

                if len(x) < max_seq_len:
                    x = x + ["[PAD]"] * (max_seq_len - len(x))
                else:
                    x = x[:max_seq_len]

                data_x.append(tokenizer.convert_tokens_to_ids(x))
                coord_x.append(compute_dist(lat1, long1, lat2, long2))
                data_y.append(int(y.strip()))

    with open(n_path, encoding="utf-8") as f:
        neigh = json.load(f)

    return data_x, coord_x, neigh, data_y
