//! Echo cancellation after the meeting: the system track is the far end (render), the
//! microphone track the capture. Both 16 kHz mono f32, already aligned by sample cursor.
//!   meetbutler-aec-bench <mic.f32> <system.f32> <out.f32> [delay_ms]   (raw little-endian f32)
use std::io::{Read, Write};
use sonora::config::EchoCanceller;
use sonora::{AudioProcessing, Config, StreamConfig};

fn read_f32(path: &str) -> Vec<f32> {
    let mut b = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (mic, sys) = (read_f32(&a[1]), read_f32(&a[2]));
    let delay: i32 = a.get(4).map(|d| d.parse().unwrap()).unwrap_or(0);
    let sc = StreamConfig::new(16_000, 1);
    let n = sc.num_frames();
    let mut apm = AudioProcessing::builder()
        .config(Config { echo_canceller: Some(EchoCanceller::default()), ..Default::default() })
        .capture_config(sc)
        .render_config(sc)
        .build();
    let t = std::time::Instant::now();
    let mut out = Vec::with_capacity(mic.len());
    let (mut r_out, mut c_out) = (vec![0f32; n], vec![0f32; n]);
    for k in 0..mic.len().min(sys.len()) / n {
        let (m, s) = (&mic[k * n..(k + 1) * n], &sys[k * n..(k + 1) * n]);
        apm.process_render_f32(&[s], &mut [&mut r_out]).unwrap();
        apm.set_stream_delay_ms(delay).unwrap();
        apm.process_capture_f32(&[m], &mut [&mut c_out]).unwrap();
        out.extend_from_slice(&c_out);
    }
    let mut f = std::fs::File::create(&a[3]).unwrap();
    for x in &out {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    eprintln!("{} s of audio in {:.1} s", mic.len() / 16000, t.elapsed().as_secs_f64());
}
