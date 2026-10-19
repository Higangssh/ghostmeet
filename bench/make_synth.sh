#!/bin/sh
# Synthesise a short Korean "meeting" with the macOS Yuna voice into bench/data/synth/
# (16 kHz mono WAV): sentences, short replies, pauses of several lengths, and one run of
# speech longer than a Silero span may be. The lines are made up for the purpose.
# Synthetic speech, so nothing here comes from a real meeting; the output stays in the
# ignored bench/data/ all the same.
set -e
out="$(dirname "$0")/data/synth"
mkdir -p "$out"
say -v Yuna -o "$out/session.aiff" "$(cat <<'TEXT'
[[slnc 1200]] 오늘은 다음 분기 출시 일정을 정리하겠습니다. [[slnc 900]]
먼저 지난주에 나온 버그 목록부터 보겠습니다. [[slnc 700]] 네. [[slnc 1500]]
설치 화면에서 버튼이 겹치는 문제는 금요일까지 고치기로 했습니다. [[slnc 400]]
그리고 로그인 오류는 서버 쪽 설정 문제로 보입니다. [[slnc 2500]]
감사합니다. [[slnc 1000]] 아니요. [[slnc 600]] 맞아요. [[slnc 3000]]
다음 안건은 문서 정리입니다 사용자 안내서와 설치 안내를 한곳에 모으고 자주 묻는 질문을 새로 쓰고
화면 캡처를 최신판으로 바꾸고 영어판과 한국어판의 차이를 맞추고 오래된 링크를 모두 고치고
검색이 잘 되도록 제목을 다듬고 예제 코드를 실제로 돌려 보고 결과를 붙여 넣는 일까지
배포 전에 한 번 더 모두가 함께 읽어 보고 빠진 곳이 없는지 확인한 다음 이번 달 안에 끝내는 것을 목표로 하겠습니다. [[slnc 1200]]
잠깐만요. [[slnc 800]] 글쎄요. [[slnc 1800]] 좋습니다 그렇게 진행하죠. [[slnc 1500]]
TEXT
)"
afconvert -f WAVE -d LEI16@16000 -c 1 "$out/session.aiff" "$out/session.wav"
rm "$out/session.aiff"
echo "$out/session.wav"
