declare -a feature=("WIENER" "HOSOYA" "INDEPENDENCE" "SECOND_EIGEN" "CIRCUIT_RANK" "SPECTRAL_RADIUS" "TRUEHOSOYA" "ZAGREB_M1" "ZAGREB_M2")

# QM9
for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with features
    python Exp/run_experiment.py -grid "Configs/Eval_QM9/gin.yaml" -dataset "QM9" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/QM9_${feature[i]}_global.json"
    done