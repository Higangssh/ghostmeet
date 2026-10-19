#!/bin/sh
# Synthesise short Korean replies with the macOS Yuna voice into bench/data/replies/
# (16 kHz mono WAV). Synthetic speech, so nothing here comes from a real meeting; the
# output stays in the ignored bench/data/ all the same.
set -e
out="$(dirname "$0")/data/replies"
mkdir -p "$out"
i=0
for text in 네 아니요 맞아요 좋습니다 그렇죠 알겠습니다 잠깐만요 글쎄요; do
  say -v Yuna -o "$out/$i.aiff" "$text"
  afconvert -f WAVE -d LEI16@16000 -c 1 "$out/$i.aiff" "$out/$i.wav"
  rm "$out/$i.aiff"
  printf '%s\t%s\n' "$i.wav" "$text"
  i=$((i + 1))
done > "$out/replies.tsv"
cat "$out/replies.tsv"
