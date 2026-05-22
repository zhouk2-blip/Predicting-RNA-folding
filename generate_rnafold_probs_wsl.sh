#!/usr/bin/env bash
set -euo pipefail

input_feats_dir="${1:?input feature directory is required}"
list_path="${2:?RNA id list path is required}"

if ! command -v RNAfold >/dev/null 2>&1; then
  echo "RNAfold is not available on PATH" >&2
  exit 1
fi

cd "$input_feats_dir"

total=$(grep -cve '^[[:space:]]*$' "$list_path")
done_count=0

while IFS= read -r raw_id || [[ -n "$raw_id" ]]; do
  rna_id="${raw_id//$'\r'/}"
  [[ -z "$rna_id" ]] && continue

  rm -f "${rna_id}_dp.ps" "${rna_id}_ss.ps"
  RNAfold -p -i "$rna_id" >/dev/null

  if [[ ! -s "${rna_id}_dp.ps" || ! -s "${rna_id}_ss.ps" ]]; then
    echo "RNAfold did not create expected files for ${rna_id}" >&2
    exit 1
  fi

  done_count=$((done_count + 1))
  if (( done_count % 50 == 0 || done_count == total )); then
    echo "RNAfold ${done_count}/${total}"
  fi
done < "$list_path"
