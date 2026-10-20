#!/usr/bin/env bash
# Fetch the public data and models of the diarization bench (run_diar.py) into
# bench/data/diar/, which is never committed, then check every file against
# bench/diar_data.sha256. Run by hand; nothing in the app or the tests calls it.
#
#   bench/fetch_diar.sh            # everything (about 2.5 GB kept, 10.3 GB streamed)
#   bench/fetch_diar.sh models ami zeroth
#
# Sources:
# - models: sherpa-onnx's GitHub releases (bench/diar_models.json lists origin and licence)
# - AMI: the Edinburgh mirror (CC BY 4.0) for audio; references and UEMs from
#   BUTSpeechFIT/AMI-diarization-setup at a pinned commit (only_words, test split)
# - Zeroth-Korean: OpenSLR 40 (CC BY 4.0). The archive is streamed and only the test set
#   and 12 training speakers are kept; the whole 10.3 GB passes through once.
#
# macOS tools (bsdtar's --include, stat -f). Downloads go to <file>.part and are moved
# into place only when complete: the length must match Content-Length and a .wav must
# start with RIFF (the AMI mirror has answered with an HTML page instead). Five tries each.
set -euo pipefail

cd "$(dirname "$0")"
D=data/diar
SUMS=$PWD/diar_data.sha256
RELEASES=https://github.com/k2-fsa/sherpa-onnx/releases/download
AMI=https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus
BUT=https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/2509d8933721023fab4def2618aabd5c28eb82e9
ZEROTH=https://www.openslr.org/resources/40/zeroth_korean.tar.gz
# training speakers for the dev meetings: never in the test set
TRAIN_SPEAKERS="124 139 151 158 164 169 183 187 194 200 207 208"

want() {  # the checksum line for a path under data/diar, if listed
  grep -E "  $1\$" "$SUMS" || true
}

good() {  # a file already in place with its listed checksum
  local line
  line=$(want "$1")
  [ -n "$line" ] && [ -f "$D/$1" ] && (cd "$D" && echo "$line" | shasum -a 256 -c --status -)
}

fetch() {  # url, path under data/diar
  local url=$1 out=$D/$2 len size try
  if good "$2"; then return 0; fi
  mkdir -p "$(dirname "$out")"
  for try in 1 2 3 4 5; do
    len=$(curl -sIL -m 60 "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' | tr -d '\r')
    if curl -sfL -m 3600 -o "$out.part" "$url"; then
      size=$(stat -f %z "$out.part")
      if [ -n "$len" ] && [ "$size" != "$len" ]; then
        echo "  $2: $size bytes, expected $len (try $try)" >&2
      elif [[ "$out" == *.wav ]] && [ "$(head -c 4 "$out.part")" != "RIFF" ]; then
        echo "  $2: not a RIFF file (try $try)" >&2
      else
        mv "$out.part" "$out"
        return 0
      fi
    fi
    sleep $((try * 5))
  done
  rm -f "$out.part"
  echo "failed: $url" >&2
  return 1
}

models() {
  local f
  if ! good models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx; then
    fetch "$RELEASES/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" models/seg.tar.bz2
    tar -xjf "$D/models/seg.tar.bz2" -C "$D/models"
    rm "$D/models/seg.tar.bz2"
  fi
  for f in 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx \
           3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx \
           3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx \
           nemo_en_titanet_small.onnx wespeaker_en_voxceleb_resnet34_LM.onnx; do
    fetch "$RELEASES/speaker-recongition-models/$f" "models/$f"  # sic: the release's tag
  done
}

ami() {
  local m k
  fetch "$BUT/lists/test.meetings.txt" ami/test.meetings.txt
  fetch "$BUT/LICENSE" ami/LICENSE.setup
  for m in $(cat "$D/ami/test.meetings.txt"); do
    fetch "$BUT/only_words/rttms/test/$m.rttm" "ami/rttm/$m.rttm"
    fetch "$BUT/uems/test/$m.uem" "ami/uem/$m.uem"
    for k in Mix-Headset Array1-01; do
      fetch "$AMI/$m/audio/$m.$k.wav" "ami/audio/$m.$k.wav"
    done
  done
}

zeroth() {
  local s args=(--include '*test_data_01/*')
  if (cd "$D" && grep "  zeroth/" "$SUMS" | shasum -a 256 -c --status -); then return 0; fi
  for s in $TRAIN_SPEAKERS; do args+=(--include "*train_data_01/003/$s/*"); done
  mkdir -p "$D/zeroth/raw" "$D/zeroth/train"
  # one pass over the archive; test speakers go to raw/, training speakers to train/
  curl -sfL --retry 5 "$ZEROTH" | tar -xz -C "$D/zeroth/raw" "${args[@]}"
  mv "$D/zeroth/raw/train_data_01" "$D/zeroth/train/" 2>/dev/null || true
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(models ami zeroth)
for s in "${steps[@]}"; do
  case $s in
    models|ami|zeroth) echo "== $s"; "$s" ;;
    *) echo "unknown step: $s" >&2; exit 2 ;;
  esac
done

echo "== checking against diar_data.sha256"
pattern=$(IFS='|'; echo "${steps[*]}")
(cd "$D" && grep -E "  ($pattern)/" "$SUMS" | shasum -a 256 -c --quiet -) && echo "all files match"
