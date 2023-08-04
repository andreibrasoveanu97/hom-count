declare -a feature=("WIENER" "HOSOYA" "SECOND_EIGEN" "CIRCUIT_RANK" "SPECTRAL_RADIUS")

# ZINC
for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_with_features_individual.yaml" -dataset "ZINC" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/ZINC_${feature[i]}_global.json"
    done