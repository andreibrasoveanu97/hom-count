declare -a feature=("WIENER" "HOSOYA" "SECOND_EIGEN" "CIRCUIT_RANK" "SPECTRAL_RADIUS" "ZAGREB_M1" "ZAGREB_M2" \
                    "ABC" "BALABAN_CYCLOMATIC" "BALABAN_J" "ECCENTRIC_DISTANCE_SUM" "ECCENTRIC" "ESTRADA" "HARARY" "HOMO_LUMO" \
                    "KIRCHOFF" "MTI" "RANDIC" "WIENER_HYPER" "WIENER_TERMINAL")

# ogbg-molhiv GIN
for i in "${!feature[@]}"
do
  echo "${feature[i]}"
    # GIN with global feature attached at the end
    python Exp/run_experiment.py -grid "Configs/Eval/gin_trafo_node.yaml" -dataset "ogbg-molhiv" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/OGBG-MOLHIV_${feature[i]}_global.json"

    # GIN with global feature attached on nodes
    python Exp/run_experiment.py -grid "Configs/Eval/gin_trafo_node_broadcast.yaml" -dataset "ogbg-molhiv" --candidates 1  --repeats 10 --graph_feat "Counts/GlobalFeatures/OGBG-MOLHIV_${feature[i]}_global.json"

    done
