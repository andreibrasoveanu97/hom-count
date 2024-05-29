declare -a feature=("WIENER" "HOSOYA" "SECOND_EIGEN" "CIRCUIT_RANK" "SPECTRAL_RADIUS" "TRUEHOSOYA"  "ZAGREB_M2" \
                    "ABC" "BALABAN_CYCLOMATIC" "BALABAN_J" "DD" "DIAMETER" "ECCENTRIC_DISTANCE_SUM" "ECCENTRIC" "ESTRADA" "HARARY" "HOMO_LUMO" \
                    "KIRCHOFF" "MTI" "RANDIC" "WIENER_HYPER" "WIENER_REVERSE" "WIENER_TERMINAL")

# ZINC-trafo
for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with global feature attached at the end
    python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_trafo_node.yaml" -dataset "ZINC" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/ZINC_${feature[i]}_global.json"

    # GIN with global feature attached on nodes
    python Exp/run_experiment.py -grid "Configs/Eval_ZINC/gin_trafo_node_broadcast.yaml" -dataset "ZINC" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/ZINC_${feature[i]}_global.json"

    done
