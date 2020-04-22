#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_OUT_DIR="${THIS_DIR}/../results"

readonly HEADER="Batch Size\tLearning Rate\tEpochs\tCompute Period\tRule Set\tADMM Iterations\tCategorical Accuracy"
readonly PATTERN='s#^.*/\([0-9\.]\+\)::Batch_Size/\([0-9\.]\+\)::Learning_Rate/\([0-9\.]\+\)::Epochs/\([0-9\.]\+\)::Compute_Period/\(single\|triple\)::Rule_Set/\([0-9\.]\+\)::ADMM_Iterations.*Categorical Accuracy: \([0-9\.]\+\)$#\1\t\2\t\3\t\4\t\5\t\6\t\7#'

function main() {
    if [[ ! $# -eq 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    trap exit SIGINT

    echo -e "${HEADER}"
    grep -R "Evaluation results for PREDICTEDNUMBER" "${BASE_OUT_DIR}" | sed "$PATTERN"
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
