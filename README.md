## Setup

1. Clone repository
```
git clone https://github.com/andreibrasoveanu97/gnn-global-features/
git checkout inter_project
cd gnn-global-features
```

2. Create and activate conda environment (this assume miniconda is installed)
```
conda create --name inter_project
conda activate inter_project
```

3. Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

4. Install PyTorch (Geometric)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
conda install openbabel fsspec rdkit graph-tool -c conda-forge
```

5. Install remaining dependencies
```
pip install -r requirements.txt
```

## Recreating experiments
Run experiments with the following scripts. Results will be in the Results directory.

**Main experiments.** Global features against no feature attached:
```
bash Scripts/GlobalFeatures_Individual/ZINC.sh
```

## Main app
**LogP predictor app** In order to run the logP inference model web app, run the following commands:

```
streamlit run main.py
```
