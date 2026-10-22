//! Can tract run segmentation-3.0 and ERes2NetV2? Loads, optimizes, runs on raw f32 files.
use std::io::{Read, Write};
use std::time::Instant;
use tract_onnx::prelude::*;
mod fbank;

fn read_f32(path: &str) -> Vec<f32> {
    let mut b = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn write_f32(path: &str, v: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn peak_mb() -> f64 {
    let mut ru: libc_rusage = unsafe { std::mem::zeroed() };
    unsafe { getrusage(0, &mut ru) };
    ru.ru_maxrss as f64 / 1048576.0 // bytes on macOS
}

#[repr(C)]
struct libc_rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    rest: [i64; 13],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut libc_rusage) -> i32;
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let t = Instant::now();
    match args[1].as_str() {
        // seg <model> <windows.f32> <n> <out.f32>
        "seg" => {
            let mut m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(&args[2])?;
            let (n, tt) = (m.symbols.sym("N"), m.symbols.sym("T"));
            m.set_input_fact(0, f32::fact(tract_onnx::prelude::tvec!(n.clone().into(), 1.to_dim(), tt.clone().into())).into())?;
            let typed = m.into_typed()?;
            let _ = (&n, &tt);
            let model = if std::env::var("TYPED").is_ok() { typed.into_runnable()? } else { typed.into_optimized()?.into_runnable()? };
            eprintln!("seg loaded {:.1}s peak {:.0} MB", t.elapsed().as_secs_f64(), peak_mb());
            let x = read_f32(&args[3]);
            let n: usize = args[4].parse().unwrap();
            let mut out = Vec::new();
            let t = Instant::now();
            for i in 0..n {
                let w = tract_ndarray::Array3::from_shape_vec((1, 1, 160000), x[i * 160000..(i + 1) * 160000].to_vec())?;
                let y = model.run(tvec!(w.into_tensor().into()))?;
                out.extend(y[0].to_plain_array_view::<f32>()?.iter().copied());
            }
            eprintln!("seg {} windows {:.2}s peak {:.0} MB", n, t.elapsed().as_secs_f64(), peak_mb());
            write_f32(&args[5], &out);
        }
        // emb <model> <feats.f32> <lens.txt> <out.f32>   feats: concatenated T x 80
        "emb" => {
            let mut m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(&args[2])?;
            if let Ok(t) = std::env::var("FIXED_T") {
                m.set_input_fact(0, f32::fact([1, t.parse::<usize>().unwrap(), 80]).into())?;
            } else {
                if std::env::var("ASSERT").is_ok() { m.symbols.add_assertion("T>=10")?; }
                let s = m.symbols.sym("T");
                m.set_input_fact(0, f32::fact(tract_onnx::prelude::tvec!(1.to_dim(), s.into(), 80.to_dim())).into())?;
            }
            let model = if std::env::var("TYPED").is_ok() { m.into_typed()?.into_runnable()? } else { m.into_optimized()?.into_runnable()? };
            eprintln!("emb loaded {:.1}s peak {:.0} MB", t.elapsed().as_secs_f64(), peak_mb());
            let x = read_f32(&args[3]);
            let lens: Vec<usize> = std::fs::read_to_string(&args[4]).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
            let (mut off, mut out) = (0, Vec::new());
            let t = Instant::now();
            for &l in &lens {
                let f = tract_ndarray::Array3::from_shape_vec((1, l, 80), x[off * 80..(off + l) * 80].to_vec())?;
                off += l;
                let y = model.run(tvec!(f.into_tensor().into()))?;
                out.extend(y[0].to_plain_array_view::<f32>()?.iter().copied());
            }
            eprintln!("emb {} inputs {} frames {:.2}s peak {:.0} MB", lens.len(), off, t.elapsed().as_secs_f64(), peak_mb());
            write_f32(&args[5], &out);
        }
        // meeting <seg> <emb> <audio.f32> <feats.f32> <lens.txt> <seg_out.f32> <emb_out.f32>
        "meeting" => {
            let mut m = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(&args[2])?;
            let (n, tt) = (m.symbols.sym("N"), m.symbols.sym("T"));
            m.set_input_fact(0, f32::fact(tract_onnx::prelude::tvec!(n.into(), 1.to_dim(), tt.into())).into())?;
            let seg = m.into_optimized()?.into_runnable()?;
            let mut e = tract_onnx::onnx().with_ignore_output_shapes(true).model_for_path(&args[3])?;
            let s = e.symbols.sym("T");
            e.set_input_fact(0, f32::fact(tract_onnx::prelude::tvec!(1.to_dim(), s.into(), 80.to_dim())).into())?;
            let emb = e.into_typed()?.into_runnable()?;
            eprintln!("loaded {:.1}s peak {:.0} MB", t.elapsed().as_secs_f64(), peak_mb());
            let audio = read_f32(&args[4]);
            let (win, shift) = (160000usize, 16000usize);
            let chunks = (audio.len() - win) / shift + 1;
            let t1 = Instant::now();
            let mut seg_out = Vec::new();
            for i in 0..chunks {
                let w = tract_ndarray::Array3::from_shape_vec((1, 1, win), audio[i * shift..i * shift + win].to_vec())?;
                let y = seg.run(tvec!(w.into_tensor().into()))?;
                seg_out.extend(y[0].to_plain_array_view::<f32>()?.iter().copied());
            }
            let seg_sec = t1.elapsed().as_secs_f64();
            eprintln!("segmentation {} windows {:.1}s peak {:.0} MB", chunks, seg_sec, peak_mb());
            drop(audio);
            let x = read_f32(&args[5]);
            let lens: Vec<usize> = std::fs::read_to_string(&args[6]).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
            let (mut off, mut out) = (0, Vec::new());
            let t2 = Instant::now();
            for &l in &lens {
                let f = tract_ndarray::Array3::from_shape_vec((1, l, 80), x[off * 80..(off + l) * 80].to_vec())?;
                off += l;
                let y = emb.run(tvec!(f.into_tensor().into()))?;
                out.extend(y[0].to_plain_array_view::<f32>()?.iter().copied());
            }
            let emb_sec = t2.elapsed().as_secs_f64();
            eprintln!("embeddings {} inputs {} frames {:.1}s peak {:.0} MB", lens.len(), off, emb_sec, peak_mb());
            println!(r#"{{"seg_sec": {:.1}, "emb_sec": {:.1}, "windows": {}, "embeddings": {}, "peak_mb": {:.0}}}"#, seg_sec, emb_sec, chunks, lens.len(), peak_mb());
            write_f32(&args[7], &seg_out);
            write_f32(&args[8], &out);
        }
        // fbank <clips.f32> <lens.txt> <out.f32>   lens in samples
        "fbank" => {
            let fb = fbank::Fbank::new();
            let x = read_f32(&args[2]);
            let lens: Vec<usize> = std::fs::read_to_string(&args[3]).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
            let (mut off, mut out) = (0, Vec::new());
            for &l in &lens {
                out.extend(fb.compute(&x[off..off + l]));
                off += l;
            }
            write_f32(&args[4], &out);
        }
        _ => panic!("seg | emb | meeting | fbank"),
    }
    Ok(())
}
