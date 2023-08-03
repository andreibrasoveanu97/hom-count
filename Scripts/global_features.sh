declare -a dataset=("ZINC")
declare -a counts_cur_path=("ZINC_full_global_6.json")

# ZINC
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"

    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10

    # GIN without features
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10


    # GCN with features
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10

    # GCN without features
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval_${dataset[i]}/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    done