#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-}"
root_dir="${2:-data}"
download_dir="$root_dir/downloads"
raw_dir="$root_dir/raw"

usage() {
  echo "Usage: $0 {ppg_dalia|wesad|ptt_ppg|bidmc|all} [data-root]" >&2
  exit 2
}

mkdir -p "$download_dir" "$raw_dir"

download_ppg_dalia() {
  mkdir -p "$raw_dir/ppg_dalia"
  curl -L --fail --retry 5 --continue-at - \
    "https://archive.ics.uci.edu/static/public/495/ppg%2Bdalia.zip" \
    -o "$download_dir/ppg_dalia.zip"
}

download_wesad() {
  mkdir -p "$raw_dir/wesad"
  curl -L --fail --retry 5 --continue-at - \
    "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download" \
    -o "$download_dir/wesad.zip"
}

download_bidmc() {
  mkdir -p "$raw_dir/bidmc"
  curl -L --fail --retry 5 --continue-at - \
    "https://physionet.org/content/bidmc/get-zip/1.0.0/" \
    -o "$download_dir/bidmc-1.0.0.zip"
  unzip -o -q "$download_dir/bidmc-1.0.0.zip" -d "$raw_dir/bidmc"
}

download_physionet_wfdb() {
  local slug="$1"
  local version="$2"
  local destination="$3"
  mkdir -p "$destination"
  wget --mirror --no-parent --no-host-directories --cut-dirs=3 \
    --continue --reject "index.html*" --reject-regex '/csv/' \
    --directory-prefix="$destination" \
    "https://physionet.org/files/$slug/$version/"
}

case "$dataset" in
  ppg_dalia) download_ppg_dalia ;;
  wesad) download_wesad ;;
  ptt_ppg) download_physionet_wfdb "pulse-transit-time-ppg" "1.1.0" "$raw_dir/ptt_ppg" ;;
  bidmc) download_bidmc ;;
  all)
    download_ppg_dalia
    download_wesad
    download_physionet_wfdb "pulse-transit-time-ppg" "1.1.0" "$raw_dir/ptt_ppg"
    download_bidmc
    ;;
  *) usage ;;
esac
