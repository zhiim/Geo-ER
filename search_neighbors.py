import json
from difflib import SequenceMatcher
from io import open
from math import atan2, cos, radians, sin, sqrt

import pandas as pd


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
        return " "

    try:
        float(lon1)
    except ValueError:
        return " "

    try:
        float(lat2)
    except ValueError:
        return " "

    try:
        float(lon2)
    except ValueError:
        return " "

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))

    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return str(round(R * c * 1000))


city = "pit"
split = "/osm_fsq/"

NEIGHBORHOOD_RADIUS = 1000
HIDDEN_SIZE = 768

csv_path_osm = city + "/osm_" + city + ".csv"
osm_dataset = pd.read_csv(csv_path_osm, index_col=0).fillna(" ")
csv_path_yelp = city + "/yelp_" + city + ".csv"
yelp_dataset = pd.read_csv(csv_path_yelp, index_col=0).fillna(" ")

dataset = osm_dataset.append(yelp_dataset, ignore_index=True)

train_path = "train_valid_test" + split + city + "/train.txt"
valid_path = "train_valid_test" + split + city + "/valid.txt"
test_path = "train_valid_test" + split + city + "/test.txt"

train_path_out = "neighborhood_train_valid_test/" + city + "/n_train.json"
valid_path_out = "neighborhood_train_valid_test/" + city + "/n_valid.json"
test_path_out = "neighborhood_train_valid_test/" + city + "/n_test.json"

for path in [train_path, valid_path, test_path]:
    entries = []

    if path == train_path:
        out_path = train_path_out
        print("Preparing train neighborhood data...")
    elif path == valid_path:
        out_path = valid_path_out
        print("Preparing valid neighborhood data...")
    else:
        out_path = test_path_out
        print("Preparing test neighborhood data...")

    count = 0
    with open(path, "r") as f:
        for line in f:
            count += 1

    c = 0
    with open(path, "r") as f:
        for line in f:
            e1 = line.split("\t")[0].lower()
            e2 = line.split("\t")[1].lower()

            entry = {}

            name1 = []
            name2 = []

            words = e1.split()

            for i, word in enumerate(words):
                if words[i - 1] == "val" and words[i - 2] == "name":
                    j = i
                    while not (
                        words[j] == "col" and words[j + 1] == "latitude"
                    ):
                        name1.append(words[j])
                        j += 1

                if words[i - 1] == "val" and words[i - 2] == "latitude":
                    lat1 = word
                if words[i - 1] == "val" and words[i - 2] == "longitude":
                    long1 = word

            words = e2.split()

            for i, word in enumerate(words):
                if words[i - 1] == "val" and words[i - 2] == "name":
                    j = i
                    while not (
                        words[j] == "col" and words[j + 1] == "latitude"
                    ):
                        name2.append(words[j])
                        j += 1

                if words[i - 1] == "val" and words[i - 2] == "latitude":
                    lat2 = word
                if words[i - 1] == "val" and words[i - 2] == "longitude":
                    long2 = word

            name1 = " ".join(name1)
            name2 = " ".join(name2)

            neighborhood1 = []
            neighborhood2 = []

            distances1 = []
            distances2 = []

            # x = tokenizer.tokenize('[CLS] ' + name1 + ' [SEP]')
            # x = tokenizer.convert_tokens_to_ids(x)
            # x = torch.tensor(x).view(1,-1)
            # x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
            # neighborhood1.append(x)
            entry["name1"] = name1

            # x = tokenizer.tokenize('[CLS] ' + name2 + ' [SEP]')
            # x = tokenizer.convert_tokens_to_ids(x)
            # x = torch.tensor(x).view(1,-1)
            # x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
            # neighborhood2.append(x)
            entry["name2"] = name2

            entry["neigh1"] = []
            entry["neigh2"] = []
            entry["dist1"] = []
            entry["dist2"] = []

            for i in range(dataset.shape[0]):
                row = dataset.iloc[i]

                dist = compute_dist(
                    lat1, long1, str(row["latitude"]), str(row["longitude"])
                )

                try:
                    dist = int(dist)
                except ValueError:
                    continue

                if (
                    jaccard_similarity(
                        name1.split(), row["name"].lower().split()
                    )
                    > 0.4
                    or similar(name1, row["name"].lower()) > 0.75
                ) and dist < NEIGHBORHOOD_RADIUS:
                    if (
                        name1 == row["name"].lower()
                        and str(row["latitude"]) == lat1
                        and str(row["longitude"]) == long1
                    ) or (
                        name2 == row["name"].lower()
                        and str(row["latitude"]) == lat2
                        and str(row["longitude"]) == long2
                    ):
                        continue

                    # x = tokenizer.tokenize('[CLS] ' + row['name'].lower() + ' [SEP]')
                    # x = tokenizer.convert_tokens_to_ids(x)
                    # x = torch.tensor(x).view(1,-1)
                    # x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
                    # neighborhood1.append(x)
                    # distances1.append(dist)
                    entry["neigh1"].append(row["name"].lower())
                    entry["dist1"].append(dist)

                dist = compute_dist(
                    lat2, long2, str(row["latitude"]), str(row["longitude"])
                )

                try:
                    dist = int(dist)
                except ValueError:
                    continue

                if (
                    jaccard_similarity(
                        name2.split(), row["name"].lower().split()
                    )
                    > 0.4
                    or similar(name2, row["name"].lower()) > 0.75
                ) and dist < NEIGHBORHOOD_RADIUS:
                    if (
                        name1 == row["name"].lower()
                        and str(row["latitude"]) == lat1
                        and str(row["longitude"]) == long1
                    ) or (
                        name2 == row["name"].lower()
                        and str(row["latitude"]) == lat2
                        and str(row["longitude"]) == long2
                    ):
                        continue

                    # x = tokenizer.tokenize('[CLS] ' + row['name'].lower() + ' [SEP]')
                    # x = tokenizer.convert_tokens_to_ids(x)
                    # x = torch.tensor(x).view(1,-1)
                    # x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
                    # neighborhood2.append(x)
                    # distances2.append(dist)
                    entry["neigh2"].append(row["name"].lower())
                    entry["dist2"].append(dist)

            # if len(neighborhood1) < 2:
            # neighborhood1.append(np.zeros(HIDDEN_SIZE))
            # distances1.append(NEIGHBORHOOD_RADIUS)

            # if len(neighborhood2) < 2:
            # neighborhood2.append(np.zeros(HIDDEN_SIZE))
            # distances2.append(NEIGHBORHOOD_RADIUS)

            # print(len(neighborhood1))
            # print(len(neighborhood1[0]))

            # time.sleep(10)

            # entry = [neighborhood1, distances1, neighborhood2, distances2]

            entries.append(entry)
            c += 1
            if c % 100 == 0:
                print(str(c / count * 100) + "%")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4)
