# MPP & CNN for object detection in remotely sensed images

This code was used to produce the results shown in:
> "CNN-based energy learning for MPP object detection in satellite images"
Jules Mabon, Mathias Ortner, Josiane Zerubia
In Proc. 2022 IEEE International Workshop on Machine Learning for Signal Processing (MLSP)
[paper](https://hal.inria.fr/view/index/docid/3715331)

> "Point process and CNN for small objects detection in satellite images"
Jules Mabon, Mathias Ortner, Josiane Zerubia
In Proc. 2022 SPIE Image and Signal Processing for Remote Sensing XXVIII
[paper (MISSING LINK)]()

If you use this code please cite our work :
> @inproceedings{mabon2022,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author = {Mabon, Jules and Ortner, Mathias and Zerubia, Josiane}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title = {{CNN}-based energy learning for {MPP} object detection in satellite images},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;booktitle = {proc. International Workshop on Machine Learning for Signal Processing ({MLSP}), {IEEE}, 2022}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year = {2022}
}

## Installation
- to compute metrics install [dota devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) in `data/` (see installation for more info)
```
cd data/
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit
cd DOTA_devkit/
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
- configure `paths_configs.json` as needed
- conda env is provided `env.yml`, setup using `conda env create -f env.yml`

## Description and usage 
We propose 2 versions of our model :
- **Manual weights model**: learns likelihood terms with CNNs, weights of energy terms (likelihood and priors) are set by hand (`model_configs/mpp/config_hrcM.json`)
- **Learned energy weights**: same as previously, but energy terms weights are learned on the dataset


### Procedure
You can skip to 3 for inference as pre-trained models are supplied.

1. Train the position and marks models
    ```
    python main.py -p train -m posnet -c config_pos.json -o
    ```
    ```
    python main.py -p train -m posnet -c config_pos.json -o
    ```
2. Set energy weights or train
    **For manual weights** :
    set model config in `models_storage/mpp/mpp_hrcM/config.json`
    **For learned weights**:
    ```
    python main.py -p train -m mpp -c config_mpp_log.json -o
    ```
3. infer on data
    ```
    python main.py -p infer -m mpp -c <model>
    ```
    with model either `mpp_hrcM` or `mpp_log`


## Project structure

```
object_detection
├── data - datasets and patch samplers
|   └── translation - translating source datasets to custom format
├── display - generic display methods
|   └── light_display - a custom pixel-perfect display toolset
├── model parts
|   ├── losses - losses for nn
|   └── unet - classical Unet architecture
├── models - see details bellow
|   ├── mpp
|   ├── shape_net
|   └── position_net
├── shapes - points, circles, rectangles
├── utils - misc functions
├── paths_configs.json - configure here the path of the datasets and where to store models
└── main.py - the main thing to run anything
```


## Data
We provide in `data_sample/DOTA_gsd50` a limited sample of the data at 0.5 m/pixel, you can download the full DOTA dataset (that contains various sources and resolutions) from https://captain-whu.github.io/DOTA/dataset.html 

We provide the code to transform the original high resolution DOTA dataset our a 0.50 m/pixel dataset:
1. setup paths in config file: `data/translation/translate_DOTA_config.json`
2. make sure data sorage paths are set as desired in `paths_config.json`
3. run 
    ```
    python main.py -p translate_dota -c data/translation/translate_DOTA_config.json
    ```


### Data structure

Each file of a dataset folder is name as `number.extension` (ie `0004.png`), files should match between folder with
their id. `/utils/data.py` provides `check_data_match` that checks if two files correspond (using the regular
expression `([0-9]+)\.[a-zA-z]+`)

```
datasets
├── DOTA_gsd50
|   ├── train
|   |   ├── raw_images
|   |   ├── images - png files
|   |   ├── raw_annotations - json files with raw annotations
|   |   ├── annotations - pikled dict, 1 is  where N is the number of objects
|   |   ├── metadata - 
|   |   └── images_w_annotations
│   └── val
|       └── ...
└── inference
   ├── DOTA_gsd50
   |  ├── train
   |  |  ├── model_1 - results on train set for model 1
   |  |  └── ...
   |  └── val
   |     ├── model_1 - results on val set for model 1
   |     └── ...
   └── ...
```

each annotation is a pickled dict with key :

- `centers`: Nx2 array of centers
- `parameters`: Nx3 array of parameters (with a,b,w : short, long, angle)
- `categories`: array of size N of strings, encoding the category of objects


## Acknowledgement

Thanks to BPI France (LiChiE contract) for funding this research work, and to the OPAL infrastructure from Université Côte d'Azur for providing computational resources and support.

