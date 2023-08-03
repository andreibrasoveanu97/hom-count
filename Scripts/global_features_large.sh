declare -a dataset=("ogbg-molhiv")
declare -a counts_cur_path=("OGBG-MOLHIV_full_global.json")

#
# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"

    # first run in each group is to generate a file for the hyperparameters tuning

    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 1  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10

    # GIN without features
#    python Exp/run_experiment.py -grid "Configs/Eval/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 1  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
#    python Exp/run_experiment.py -grid "Configs/Eval/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
#    python Exp/run_experiment.py -grid "Configs/Eval/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10

    # GCN with features
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 1  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10

    # GCN without features
#    python Exp/run_experiment.py -grid "Configs/Eval/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 1  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
#    python Exp/run_experiment.py -grid "Configs/Eval/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/GlobalFeatures/${counts_cur_path[i]}"
#    python Exp/run_experiment.py -grid "Configs/Eval/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    done

