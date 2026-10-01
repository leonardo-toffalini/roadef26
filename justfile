# Check a solution. `just check data/toy/toy sats` prints loads, largest first.
check instance mode="":
    #!/usr/bin/env bash
    set -euo pipefail
    mode="{{mode}}"
    if [[ "$mode" == "sats" ]]; then
        ./check --instance "{{instance}}" | jq '[.saturations[].sat] | sort | reverse'
    elif [[ "$mode" == "pretty" ]]; then
        ./check --instance "{{instance}}" --pretty
    else
        ./check --instance "{{instance}}" $mode
    fi
