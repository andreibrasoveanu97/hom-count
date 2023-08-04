declare -a feature=("HOSOYA" "SECOND_EIGEN" "CIRCUIT_RANK" "SPECTRAL_RADIUS" "DUMMY")

# ogbg-molhiv
for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval/gin_with_features_individual.yaml" -dataset "ogbg-molhiv" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/OGBG-MOLHIV_${feature[i]}_global.json"
    done