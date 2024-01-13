# Computer Vision Challenge for the Technical Interview of HUK Coburg
The analysis of the target labels, and the evaluation and comparison of different models can be found in [eda_and_plots.ipynb](./eda_and_plots.ipynb).

## Usage
After installing the required libraries, a single model can be trained by calling `train.py` with `config.json` where the specific hyperparameters of the model are specified: 
```shell
python train.py config.json
```

Equivalently `optimize.py` can be called to perform a hyperparameter optimization using the Hyperband algorithm:
```shell
python optimize.py config.json <experiment_name>
```
