# GlobalFeaturesGNNs
Within this repository, you will find the code belonging to the GNN section of the submission titled **[Extending Graph Neural Networks with Global Features](https://openreview.net/forum?id=aisVQy6R2k)** (LOG, 2023).

## Setup

1. Clone repository
```
git clone https://github.com/andreibrasoveanu97/hom-count/
cd hom-count
```

2. Create and activate conda environment (this assume miniconda is installed)
```
conda create --name HOM
conda activate HOM
```

3. Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

4. Install PyTorch (Geometric)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
```

5. Install remaining dependencies
```
pip install -r requirements.txt
```

## Recreating experiments
Run experimentes with the following scripts. Results will be in the Results directory.

**Main experiments.** Global features against no feature attached:
```
bash Scripts/GlobalFeatures_Individual/ZINC.sh
bash Scripts/GlobalFeatures_Individual/ogbg-molhiv.sh
```

**Ablation.** Impact of random noise in global features:
```
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_with_features_individual.yaml" -dataset "ZINC" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/ZINC_DUMMY_global.json"
python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features_individual.yaml" -dataset "ogbg-molhiv" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/OGBG-MOLHIV_DUMMY_global.json"
```
