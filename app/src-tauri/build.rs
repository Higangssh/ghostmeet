fn main() {
    // the Swift runtime for ScreenCaptureKit (see crates/meetbutler-audio/build.rs)
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
    tauri_build::build()
}
