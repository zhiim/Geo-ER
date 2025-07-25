# Geo-ER
### Code for 'Geospatial Entity Resolution' paper (WWW 2022)

Geo-ER is an Entity Resolution (ER) framework to match geospatial entities. It allows to integrate geospatial databases, performing deduplication of the entries. Geo-ER leverages Transformer-based Language Models (LMs), Distance embedding and a novel Neighbourhood Attention component, based on Graph Attention (GAT).

### <ins>UPDATE</ins>
Now Geo-ER supports JSON-formatted neighbors for entities, to make it easier to use it for other tasks and dataset. The JSON format is the following:

```
# list of python dictionaries
[
    {
        "name1": "...",               # name of entity1
        "name2": "...",               # name of entity2
        "neigh1": [
                                      # list of entity1 neighbors' names
        ],
        "neigh2": [
                                      # list of entity2 neighbors' names
        ],
        "dist1": [
                                      # list of entity1 neighbors' distances from entity1
        ],
        "dist2": [
                                      # list of entity2 neighbors' distances from entity2
        ]
    }
]
```

### Requirements

* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers 4.9.2

Install required packages
```
pip install -r requirements.txt
```


### Geospatial ER

The following image (left) depicts an example of Geospatial ER, in which two sources are being joined. Geo-ER uses textual information, distance and spatial neighbours information (right) to infer if two records, from the two sources, refer to the same real-world entity.

<img src="imgs/geo_er_examples.jpg" alt="Example of geospatial ER" width="62%"/><img src="imgs/neighbourhood.jpg" alt="Example of geospatial ER" width="38%"/>

Each entity is pre-serialized as follows:

```
COL name VAL Wine and Spirits COL latitude VAL 40.4535 COL longitude VAL -80.009 COL address VAL Ohio Street COL postalCode VAL NULL
```

Each entity pair ``<e_1, e_2>`` is serialized as follows:

```
<e_1> \t <e_2> \t <label>
```
where ``<e_i>`` is the serialized version of entity ``i`` and ``<label>`` is the either ``0`` (no-match) or ``1`` (match).


### Training

To train Geo-ER, please use:

```
python main.py -c pit -s osm_fsq
```

The meaning of the flags and their possible values are listed here:
* ``-c``, ``--city``: Specify the dataset of which city you wish to use for training. Possible values are ``sin``, ``edi``, ``tor``, ``pit``.
* ``-s``, ``--source``: Specify the sources to be joined to create the dataset. Possible values are ``osm_fsq``, ``osm_yelp``.


### Datasets

Please use the ``search_neighbors.py`` script to generate the neighbourhood data for any city. The files containing such neighbourhood information are quite large, being composed of BERT encodings.

This paper introduces 8 real-world datasets, joining 3 different sources (``OpenStreetMap``, ``Foursquare`` and ``Yelp``), in 4 different cities (``Singapore``, ``Edinburgh``, ``Toronto`` and ``Pittsburgh``). The statistics of the datasets are the following:



| Source          | City      | Size     | Positive (%)  |
|-----------------|-----------|----------|---------------|
|OSM-FSQ          | Singapore | 19,243   | 2,116 (11.0%) |
|                 | Edinburgh | 17,386   | 3,350 (19.3%) |
|                 | Toronto   | 17,858   | 3,862 (21.6%) |
|                 | Pittsburgh| 5,001    | 1,454 (29.1%) |
|||||
|OSM-Yelp         | Singapore | 21,588   | 2,941 (13.6%) |
|                 | Edinburgh | 18,733   | 2,310 (12.3%) |
|                 | Toronto   | 27,969   | 5,426 (19.4%) |
|                 | Pittsburgh| 5,116    | 1,622 (31.7%) |

The column ``Positive (%)`` shows the number of positive samples.
The 8 datasets can be found in ``data/train_valid_test/``

#### If you wish to use any of our datasets for your research, please cite our paper

### Citation Information
```
@inproceedings{10.1145/3485447.3512026,
  author = {Balsebre, Pasquale and Yao, Dezhong and Cong, Gao and Hai, Zhen},
  title = {Geospatial Entity Resolution},
  year = {2022},
  isbn = {9781450390965},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3485447.3512026},
  booktitle = {Proceedings of the ACM Web Conference 2022},
  location = {Virtual Event, Lyon, France},
  series = {WWW '22}
}
```
