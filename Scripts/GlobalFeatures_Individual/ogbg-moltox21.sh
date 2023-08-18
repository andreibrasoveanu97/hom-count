declare -a feature=("WIENER" "HOSOYA" "INDEPENDENCE" "CIRCUIT_RANK" "SPECTRAL_RADIUS" "ZAGREB_M1" "ZAGREB_M2")

# ogbg-moltox21
for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features_individual.yaml" -dataset "ogbg-moltox21" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/OGBG-MOLTOX21_${feature[i]}_global.json"
    done