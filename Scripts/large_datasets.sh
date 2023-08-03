declare -a dataset=("ogbg-molhiv")
declare -a counts_cur_path=("OGBG-MOLHIV_full_kernel_max_50_run1.singleton_filtered")


# ogb
for i in "${!dataset[@]}"
do
    echo "${dataset[i]}"

    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/${counts_cur_path[i]}"
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gin_with_features.yaml" -dataset "${dataset[i]}" --repeats 2 --nr_diff_counts 9
    
    # GIN without features
    python Exp/run_experiment.py -grid "Configs/Eval/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/${counts_cur_path[i]}"    
    python Exp/run_experiment.py -grid "Configs/Eval/gin_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gin_without_features.yaml" -dataset "${dataset[i]}" --repeats 2 --nr_diff_counts 9

    # GCN with features
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/${counts_cur_path[i]}"    
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gcn_with_features.yaml" -dataset "${dataset[i]}" --repeats 2 --nr_diff_counts 9

    # GCN without features
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/${counts_cur_path[i]}"    
    python Exp/run_experiment.py -grid "Configs/Eval/gcn_without_features.yaml" -dataset "${dataset[i]}" --candidates 50  --repeats 10
    python Exp/run_multi_count_experiment.py -grid "Configs/Eval/gcn_without_features.yaml" -dataset "${dataset[i]}" --repeats 2 --nr_diff_counts 9
    done

# ZINC

# GIN with features
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_with_features.yaml" -dataset "ZINC" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/ZINC_SUBSET_full_kernel_max_50_run1.singleton_filtered"    
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_with_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10
python Exp/run_multi_count_experiment.py -grid "Configs/Eval_ZINC/gin_with_features.yaml" -dataset "ZINC" --repeats 2 --nr_diff_counts 9

# GIN without features
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/ZINC_SUBSET_full_kernel_max_50_run1.singleton_filtered"    
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10
python Exp/run_multi_count_experiment.py -grid "Configs/Eval_ZINC/gin_without_features.yaml" -dataset "ZINC" --repeats 2 --nr_diff_counts 9

# GCN with features
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_with_features.yaml" -dataset "ZINC" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/ZINC_SUBSET_full_kernel_max_50_run1.singleton_filtered"    
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_with_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10
python Exp/run_multi_count_experiment.py -grid "Configs/Eval_ZINC/gcn_with_features.yaml" -dataset "ZINC" --repeats 2 --nr_diff_counts 9

# GCN without features
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 1 --graph_feat "Counts/RepeatedCounts/ZINC_SUBSET_full_kernel_max_50_run1.singleton_filtered"    
python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gcn_without_features.yaml" -dataset "ZINC" --candidates 50  --repeats 10
python Exp/run_multi_count_experiment.py -grid "Configs/Eval_ZINC/gcn_without_features.yaml" -dataset "ZINC" --repeats 2 --nr_diff_counts 9

