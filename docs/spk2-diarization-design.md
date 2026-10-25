# SPK-2 diarization in Rust (0.3 design memo)

The evidence is `bench/results/RESULTS.md` §9 (the chunked prototype's dev and test). The bench is a Python prototype. Numbers below that must be measured again in the app say so.

## Scope

- Split several people in an in-person meeting (one microphone channel) and several people in a call's far-side channel (system).
- In a call, the microphone is "me" as a whole, as SPK-1 says. This relies on echo cancellation (REC-4, `docs/rec4-aec-research.md`).
- It runs after the meeting. Live captions get no speaker. Live speakers relabelled 40% or more, so they are off by default (§9).
- The result goes into each segment's `speaker` field. Naming and merging speakers is SPK-3.

## Models

| | segmentation-3.0 | ERes2NetV2 (zh-cn 16k common) |
|---|---|---|
| Source | pyannote, ONNX redistributed on GitHub by sherpa-onnx (not gated) | 3D-Speaker (modelscope `iic/speech_eres2netv2_sv_zh-cn_16k-common`), redistributed by sherpa-onnx |
| Licence | MIT | Apache-2.0 |
| File / sha256 | `model.onnx` 5,992,913 B / `220ad67c…1079` | 71,441,526 B / `bf1a75b9…2239` |
| Input → output | waveform (N, 1, 160000) → (N, 589, 7) powerset | fbank (N, T, 80) → (N, 192) |

ERes2NetV2's ONNX metadata is `normalize_samples=1`, `feature_normalize_type=global-mean`. The app must implement a Kaldi-compatible 80-dim fbank and the per-frame mean subtraction itself. The bench used sherpa's C++ implementation (kaldi-native-fbank). **The first risk is whether a Rust fbank gives the same values as sherpa.** The slow tier's pass bar is a cosine of 0.999 or more against sherpa's embedding for the same input.

The user downloads the models (the app never downloads on its own). They are checked by sha256 and loaded once per process in `ModelCache`.

## Pipeline

1. **Segmentation.**
   - Read 10 s windows every 1 s from `PcmStore`. The audio is never held in memory whole.
   - Each window gives labels (int8) for 589 frames × 3 local speakers. 4 hours is 14.4k windows, 24 MB.
2. **Embeddings.**
   - For each (window, local speaker), make one embedding from the samples where that speaker talks alone.
   - A 4-hour room meeting has about 26k, at 192 × f32 = 20 MB.
   - Those with under 10 frames, which get no embedding, follow the tail rule below.
3. **Clustering inside a piece.**
   - 1200 s pieces (+60 s overlap), cosine average linkage, cut 0.15, piece floor 0 (§9, preregistered).
   - About 2.3k embeddings per piece; the condensed distances are 2.6M doubles (21 MB).
4. **Two-stage link (differs from the bench).**
   - The bench put every in-piece cluster up as a global candidate. A 4-hour room meeting had 6–7 thousand, which is why it went over 1.0 GB.
   - The app puts up only **K representatives per piece (proposed: 32)**, largest speech seconds first. The rest attach to the representative in the same piece whose centroid is nearest.
   - The global link is average linkage over the representatives' centroids, cut 0.3. Where the overlap says two are the same person, they are joined first (must-link).
   - 4 hours gives at most 13 × 32 = 416 candidates. It grows with meeting length, not with its square.
   - There is a risk. If a quiet person's cluster does not make the top K in a piece, it attaches to someone else's representative. The vanished gate (special speakers) checks this.
   - K is a new value the bench did not have. **It goes through choosing on bench dev and checking on test once more.** Report the difference (DER, vanished) from the preregistered setting.
5. **Global floor and speaker cap.**
   - A linked speaker whose total speech is under 1% of the meeting attaches to the nearest large speaker. If no speaker passes, the largest one stays.
   - If more than 20 remain after that, keep the 20 largest and attach the rest to the nearest speaker.
   - Over-counting (7 speakers in a 2-person meeting) is plain to the user. Over-splitting left by the cap is fixed by the user with SPK-3's "merge two speakers". Design the two together.
6. **Turn reconstruction.**
   - For each frame, pick as many of the most-voted clusters as there are local speakers talking in that frame.
   - Computed in blocks (bench 7f21b5c). No frame × cluster matrix is built, however many clusters there are.
   - min_on 0.3 s, min_off 0.5 s.

## Determinism rules

- Clusters are numbered in the order their first embedding appears (first-seen). On a tied vote the lower number wins. In the bench, the numbering rule alone moved DER by 2 points.
- A tail without an embedding goes to the neighbouring speaker in time that overlaps that span most. With no neighbour it is dropped.
- Every sort is stable. Distances are computed as in the bench: an f32 product converted to f64, in the same order. An f32 matrix product split into blocks rounds differently (found in 86b3c1d).
- The same input must give the same output. A default-tier test pins this.

## Memory budget (4 hours, target 1 GB)

Values measured in one Python process in the bench's long step, and what the app should expect. They depend on the app's ORT settings (arena, threads), so measure again in the app.

The bench figures are the growth in current RSS at each stage, on a 4-hour room meeting (26k embeddings). Python, sherpa-onnx 1.13.8, ORT 1.31, 4 threads.

| Item | Bench (ERes2NetV2) | For reference: CAM++ | App expectation |
|---|---:|---:|---|
| Runtime (Python, numpy, sherpa imports) | 31 MB | 32 MB | The app is already running. Not counted separately |
| Segmentation session | 37 MB | 36 MB | Similar |
| Embedding model weights (file 71 MB / 28 MB) | 142 MB | 87 MB | About twice the file. ORT seems to keep a separate optimised copy (estimated) |
| **ORT arena and working memory (embeddings)** | **about 545 MB** | about 105 MB | Fills within the first 100 embeddings (498 → 769 → 816 MB). The arena seems to hold the working space of the longest input (10 s, about 1,000 fbank frames) (estimated). See below |
| Audio buffer | about 0 | about 0 | The file is read a little at a time. 0.6 MB per window |
| Segmentation labels (int8) | 24 MB (+about 25 of segmentation work) | Same | 24 MB. The bench makes one more copy when stacking (+26). The app writes straight into one buffer |
| Embedding store (26k × 192 f32) | 19 MB (+list overhead, about 26 of remaining growth) | Same | 20 MB |
| (Window, speaker) pairs and span lists | 10 MB | 10 MB | A few MB |
| Clustering | about 40 MB inside pieces, about 160 MB with global (fixmem, 872 → 1033 MB after extraction) | | With the two-stage link, global is a few MB and in-piece about 40 MB |
| Turn reconstruction | 22 MB | | Same |
| **Total (peak RSS)** | 870 MB after extraction, 1046 MB overall (fixmem) | 411 MB, 1064 MB | About 600 MB calling ORT directly (estimated; in the check below, ERes2NetV2 on ORT called directly peaked at 438 MB) |

Most of ERes2NetV2's memory is ORT's working memory, not the weights. Three knobs might reduce it in the app (Rust `ort`). All are estimated and must be measured.

1. ~~Turn the arena off, or grow it only as requested~~: no effect in the check. Turning it off added 200 MB (below).
2. Cap the embedding input length (for example 3–5 s). Working memory grows with input length. But the embeddings change, so DER must be measured again in the bench.
3. Thread count. Check whether each thread has its own working space.

## When it runs

Bench Python processing time for a 4-hour meeting is 25–37 min for ERes2NetV2 (4 threads) and 9–10 min for CAM++. The final pass (large-v3-turbo, Metal) is about 22.6 min.

- **A. Right after the final pass (proposed default).**
  - Memory does not add up. The budget above assumes this.
  - After-meeting processing for a 4-hour meeting is about 48–60 min. It needs progress and cancel. Cancelling leaves the transcript without speakers.
- **B. Overlapping the final pass.**
  - whisper is on Metal (GPU) and diarization on the CPU, so there is room to overlap.
  - Memory adds up. CPU contention (whisper's CPU part, the decoder) was not measured.
- **C. Segmentation and embeddings each time a piece ends during the meeting.**
  - A 1200 s piece closes during the meeting. If closed pieces are segmented and embedded with spare CPU during the meeting, only the global link and reconstruction (seconds) are left afterwards.
  - It fits the rule "never re-process the whole session during the meeting", since each piece is processed once.
  - The risk is CPU contention with live captions. The caption delay (STT target) must hold.
- Proposal for 0.3: start with A, and look at C after measuring. Neither B nor C is measured yet.

## Test plan

- **Default tier (no model, no network, fast).**
  - Fake segmentation labels and fake embeddings (one direction per speaker + noise). Port the cases from the bench's `tests/test_diar_chunk.py`.
  - Cases to port:
    - a late joiner survives inside a piece
    - someone who speaks a little in each piece survives in total
    - the same voice is one speaker
    - overlap joins a drift across a boundary
    - first-seen numbering, and the same input gives the same output
  - New cases:
    - the cap of K representatives
    - the 20-speaker cap (keeps the largest when over)
    - no speaker passes the floor
    - a tail without an embedding
    - with thousands of candidates, the global step's allocation is bounded by K × number of pieces
  - For fbank, keep reference values made from synthetic signals (sine, silence) as a fixture. The fixture holds derived numbers only.
- **Slow tier (`--ignored`, models from `MEETBUTLER_MODELS`, no downloads).**
  - Uses the bench's synthetic Korean meetings and AMI (`bench/data/`, outside the repo).
  - What to check:
    1. the cosine between Rust and sherpa embeddings of the same audio
    2. the Rust result's DER within 0.5 points of the Python baseline
    3. peak RSS and processing time for a 4-hour meeting
  - Add a bench step that reads the Rust results, so the same scorer measures them.
- Real meeting audio is never used in tests or fixtures.

## New crate candidates

| crate | Use | Licence | Note |
|---|---|---|---|
| `ort` 2.0.0-rc.x | ONNX Runtime binding | MIT OR Apache-2.0 | 2.0 is still an RC. The default build downloads an ORT binary at build time. **Decision: turn the download feature off and ship a library pinned by version and sha256 in the bundle with `load-dynamic`** (Decision below). Add that check to `security.rs`. |
| ONNX Runtime | Runtime | MIT | May add tens of MB to the dmg (50 MB target, must be measured). |
| `ndarray` | Tensors | MIT OR Apache-2.0 | Used by `ort`. |
| `realfft` / `rustfft` | FFT for fbank | MIT OR Apache-2.0 | fbank is our own implementation, compared with sherpa. |
| `kodama` | Hierarchical clustering (average, NN-chain) | MIT | For clustering inside pieces. The cut rule ("merge only below the threshold") and tie handling must match the bench. Our own implementation is also possible. |
| (existing) `tract-onnx` | Already used for Silero | MIT OR Apache-2.0 | Pure Rust, so no binary. Op support and speed for the two models were not checked. If it works, no new runtime is needed. Worth checking first. |

## Alternative: a sherpa-onnx C API binding

Call sherpa-onnx's (Apache-2.0) C API from Rust (for example `sherpa-rs`, MIT).

For:
- The same fbank, segmentation and embedding path as the bench. The reproduction risk (fbank differences) goes away.
- No model input/output handling to write.

Against:
- sherpa's offline diarization clusters the whole meeting at once. That is N² memory over 4 hours, 5–6 GB in the bench. So the clustering has to be ours anyway. Only the segmentation and the embedding extractor would be used.
- The C++ build (CMake) and bundled ORT must go into the Windows cross build and CI. Dependencies grow and so does the dmg.
- It depends on how well the binding crate is maintained.

Judgement: call segmentation and embeddings directly with `ort` (Decision below). If the embeddings miss the slow-tier bar, replace only the extractor with the sherpa binding. Clustering and reconstruction are app code either way.

## Decision (2026-10-22)

- **Both models run on `ort`.** Keeping only segmentation on tract would mean two runtimes. The existing Silero (tract) stays as it is.
- **Supply chain:**
  - `ort` does not fetch an ORT binary at build time (download feature off).
  - An ONNX Runtime dynamic library pinned by version and sha256 ships in the bundle via `load-dynamic`.
  - The library is fetched by a script a person runs, like `bench/fetch_diar.sh`, and checked by hash.
  - `app/src-tauri/tests/security.rs` gets a check that "`ort`'s download feature is off".
- **Order:** after 0.2 closes, this skeleton (build, bundle, check) is the first commit of the 0.3 implementation. Until then only this design changes.
- fbank is our own implementation (`rustfft`), with no direct comparison against kaldi-native-fbank. The end-to-end embedding cosine (0.9999998 or more) is enough.

## Checks: tract, the ort arena, fbank (2026-10-22)

No app code changed. The probe crate is `bench/tract_probe/` (outside the app workspace, build output in `bench/data/`). Inputs are the bench's Korean test meetings (synthetic, public speech clips). Machine load swung during the checks (load average 2–34 from other processes). Read times as relative values from the same inputs measured alternately.

**tract-onnx 0.23.9 (latest stable)**

- **segmentation-3.0: loads.**
  - Given a fixed input shape (1, 1, 160000), it fails at inference with "Unifying shapes 1,1,160000 and N,1,T".
  - Keeping the model's own symbols (N, 1, T) and running with `with_ignore_output_shapes(true)` and `into_optimized()` works.
  - Output differs from onnxruntime by at most 2.3e-4, and frame labels are 99.9999% the same over 1,796 windows of a 30-minute meeting (1 frame in about a million).
  - 51 ms per window on one thread. 92 s for a 30-minute meeting, about 12 min for 4 hours.
- **ERes2NetV2: correct, but does not optimise.**
  - `into_optimized()` fails in declutter: `Pulling #262 ".../layer2/layer2.0/conv1/Conv.downsample..." Downsample over #261 "...conv3/Conv.einsum.concat-einsum-I.0..26.unmerge_out" Reshape: Invalid axis`. The blocked op is **the transform that pulls a Downsample over a Reshape**.
  - Asserting a lower bound on T does not help; fixing T lets it optimise. That means recompiling for each length (about 0.2 s), which is hard to use.
  - Running unoptimised (`into_typed()`) with symbolic T works. All 3,104 embeddings of a 30-minute meeting have a cosine of 0.99999976 or more against sherpa.
  - It is slow:
    - On the same 40 inputs (23,056 frames): tract typed 14.7 s, onnxruntime one thread 9.5 s, four threads 5.0 s.
    - A 30-minute meeting is 92 s segmentation + 940 s embeddings, RTF 0.57 (one thread). 4 hours would take over 2 hours.
- **Judgement: tract alone does not work.** tract is enough for segmentation. ERes2NetV2 needs ort until tract fixes this optimisation bug. If ort comes in, running both models on ort is simpler.
  - Alternative one: cut ERes2NetV2 inputs to a few fixed lengths and cache an optimised plan per length. The embeddings change, so it must be measured again in the bench.
  - Alternative two: report the bug to tract. It is an external post, so it needs the user's approval, and the schedule does not need it, so it is not done. If the user wants it, use the repro steps below.
  - **Repro steps:**
    1. Fetch `3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx` (sha256 `bf1a75b9…2239`) with `bench/fetch_diar.sh models`.
    2. Build `bench/tract_probe` (tract-onnx 0.23.9).
    3. Run `meetbutler-tract-probe emb <model> <feats.f32> <lens.txt> <out.f32>`. The input fact is `(1, T, 80)`, T symbolic, `into_optimized()`.
    4. It fails in declutter with `Downsample over ... Reshape: Invalid axis`. Fixing T with `FIXED_T=536` passes, and so does `TYPED=1` (no optimisation).
    - The report carries the tract version, the model's source URL and sha256, the failure message, and the difference between the three cases above. The failure happens while loading the model, so no input values are needed. No meeting audio or features go in it.

**onnxruntime 1.31 arena (ERes2NetV2, the same 40 inputs, a separate process per setting)**

| Arena | Threads | After load | Peak RSS | Time |
|---|---:|---:|---:|---:|
| Default | 1 | 154 MB | 438 MB | 9.45 s |
| Off (`enable_cpu_mem_arena=false`) | 1 | 155 MB | **653 MB** | 9.58 s |
| `kSameAsRequested` | 1 | 155 MB | 438 MB | 9.59 s |
| Default | 4 | 178 MB | 467 MB | 4.97 s |
| Off | 4 | 178 MB | **666 MB** | 5.10 s |
| `kSameAsRequested` | 4 | 179 MB | 464 MB | 5.06 s |

- Turning the arena off raised the peak by 200 MB. It seems allocations are not reused and freed memory is not returned to the OS (estimated). `kSameAsRequested` makes no difference. **Turning the arena off is not a knob.**
- ORT called directly uses about 280 MB of working memory (438 − 155). That is less than the 545 MB of the bench long step through sherpa. The cause of the gap (sherpa's session settings, 4 threads, sharing the process with the segmentation session) was not checked.
- The remaining knob is a cap on embedding input length (see the budget table above). The app's ERes2NetV2 part is taken as about 450 MB, and about 600 MB with segmentation and clustering (estimated, measure again in the app).

**fbank**

- Candidate: **our own implementation**. The FFT is `rustfft` 6.4.1 (MIT OR Apache-2.0), already in the workspace. No new crate.
- A numpy reference of sherpa's defaults (snip_edges off, 20 Hz to Nyquist−400 Hz, 80 bins, Povey window, 0.97 pre-emphasis, DC removal, samples in [−1, 1]) gives the same embeddings as sherpa. On 40 clips (0.5–10 s, room and call), the ORT embeddings have a cosine of 0.9999998 or more against sherpa.
- The Rust implementation (about 100 lines) was compared frame by frame with the numpy reference over 23,256 frames (including silence and a 440 Hz sine). The log-mel difference is at most 3.6e-4 (a bin with almost no energy, −14.8) and 2.9e-6 on average. Rust fbank → ORT embeddings have a cosine of 0.99999982 or more against sherpa.
- No direct comparison against kaldi-native-fbank itself (Decision). Instead, it was matched end to end through the embeddings of sherpa, which uses kaldi-native-fbank.
- App tests take this comparison as a fixture. Only the fbank values of synthetic signals (derived numbers) go in.
