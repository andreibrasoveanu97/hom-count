declare -a feature=("WIENER" "HOSOYA" "INDEPENDENCE" "SECOND_EIGEN" "CIRCUIT_RANK" "SPECTRAL_RADIUS" "ZAGREB_M1" "ZAGREB_M2")

for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval_PEPTIDES/gin.yaml" -dataset "PEPTIDES" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/PEPTIDES_${feature[i]}_global.json"
    done