// ScreenCaptureKit's bindings are Swift: anything linking them (this crate's tests, the
// app) has to find the Swift runtime, which lives in /usr/lib/swift on macOS 12.3+.
// A dependency's link arguments do not reach its dependents, so each binary adds this.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
