//! Local-only posture of the desktop app.
//!
//! The Python backend's security tests were about its HTTP server (loopback bind, CORS).
//! The app has no server at all, so these replace them: no port is opened, the window
//! loads only the app's own files under a strict CSP, and nothing in the build talks to
//! the network or reports home.

use std::path::Path;

fn read(rel: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).unwrap()
}

fn config() -> serde_json::Value {
    serde_json::from_str(&read("tauri.conf.json")).unwrap()
}

fn csp() -> String {
    config()["app"]["security"]["csp"]
        .as_str()
        .expect("a CSP is set")
        .to_string()
}

fn directive(name: &str) -> String {
    csp()
        .split(';')
        .map(str::trim)
        .find(|d| d.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("CSP has no {name}"))
        .to_string()
}

/// Packages compiled into the app for a desktop target, from `cargo tree` (offline: the
/// lockfile already pins everything). Cargo.lock alone is not enough: it also lists
/// crates only mobile targets build, such as Tauri's iOS/Android asset client.
fn packages_for(package: &str, features: &str, target: &str) -> Vec<String> {
    let out = std::process::Command::new(env!("CARGO"))
        .args([
            "tree",
            "--offline",
            "-p",
            package,
            "--features",
            features,
            "-e",
            "normal,build",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ])
        .args(["--target", target])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run cargo tree");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

#[test]
fn the_window_loads_only_the_apps_own_files() {
    let build = &config()["build"];
    assert_eq!(build["frontendDist"], "../ui", "bundled files, not a URL");
    assert!(
        build.get("devUrl").is_none(),
        "no dev server to point the window at"
    );
    for w in config()["app"]["windows"].as_array().unwrap() {
        assert!(
            w.get("url")
                .is_none_or(|u| !u.as_str().unwrap().contains("://")),
            "{w}"
        );
    }
}

#[test]
fn the_csp_allows_no_remote_code_or_connections() {
    assert_eq!(directive("default-src"), "default-src 'self'");
    assert_eq!(directive("script-src"), "script-src 'self'");
    assert_eq!(directive("object-src"), "object-src 'none'");
    // IPC to the Rust side, and nothing else
    assert_eq!(
        directive("connect-src"),
        "connect-src ipc: http://ipc.localhost"
    );
}

#[test]
fn the_csp_never_allows_eval_or_inline_script() {
    let csp = csp();
    let script = directive("script-src");
    for bad in [
        "'unsafe-eval'",
        "'unsafe-inline'",
        "'wasm-unsafe-eval'",
        "*",
        "https:",
        "http:",
        "data:",
    ] {
        assert!(
            !script.split_whitespace().any(|t| t == bad),
            "script-src allows {bad}"
        );
    }
    assert!(!csp.contains("'unsafe-eval'"));
    assert!(
        !csp.split_whitespace().any(|t| t == "*"),
        "a wildcard source: {csp}"
    );
}

/// Every file the windows load, by extension.
fn ui_files(ext: &str) -> Vec<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ui");
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(ext))
        .map(|n| format!("../ui/{n}"))
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no {ext} files");
    out
}

#[test]
fn no_page_has_inline_script_to_need_it() {
    let scripts: Vec<String> = ui_files(".js")
        .iter()
        .map(|f| f.trim_start_matches("../ui/").to_string())
        .collect();
    for page in ui_files(".html") {
        let html = read(&page);
        assert!(!html.contains("<script>"), "inline script in {page}");
        for attr in [
            "onclick=",
            "onload=",
            "onerror=",
            "onchange=",
            "oninput=",
            "onsubmit=",
            "javascript:",
        ] {
            assert!(!html.contains(attr), "{page} has {attr}");
        }
        for line in html.lines().filter(|l| l.contains("<script")) {
            let src = line
                .split("src=\"")
                .nth(1)
                .and_then(|r| r.split('"').next());
            assert!(
                src.is_some_and(|s| scripts.iter().any(|f| f == s)),
                "{page}: a script that is not one of the app's own files: {line}"
            );
        }
    }
}

#[test]
fn the_window_gets_only_the_core_permissions() {
    let caps: serde_json::Value = serde_json::from_str(&read("capabilities/default.json")).unwrap();
    assert_eq!(caps["permissions"], serde_json::json!(["core:default"]));
    assert!(
        caps.get("remote").is_none(),
        "no remote URL may call into the app"
    );
}

#[test]
fn nothing_in_the_build_can_serve_a_port_or_reach_the_network() {
    for target in [
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-pc-windows-msvc",
    ] {
        let packages = packages_for("meetbutler", "", target);
        assert!(
            packages.iter().any(|p| p == "tauri"),
            "{target}: cargo tree listed nothing"
        );
        assert_no_network(target, &packages);
    }
}

#[test]
fn the_summary_sidecar_cannot_serve_a_port_or_reach_the_network_either() {
    // it ships beside the app and reads meeting transcripts, so the same rules apply
    for target in [
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-pc-windows-msvc",
    ] {
        let packages = packages_for("meetbutler-summarizer", "llama", target);
        assert!(
            packages.iter().any(|p| p == "llama-cpp-2"),
            "{target}: the llama backend is not in the tree"
        );
        assert_no_network(target, &packages);
    }
}

#[test]
fn the_summary_sidecar_never_binds_a_socket() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/meetbutler-summarizer/src");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let src = std::fs::read_to_string(&path).unwrap();
        for needle in ["TcpListener", "UdpSocket", "TcpStream", "std::net"] {
            assert!(!src.contains(needle), "{} uses {needle}", path.display());
        }
    }
}

fn assert_no_network(target: &str, packages: &[String]) {
    for banned in [
        // servers
        "hyper",
        "axum",
        "actix-web",
        "warp",
        "tiny_http",
        "rouille",
        "tauri-plugin-localhost",
        // clients and plugins that fetch or open things
        "reqwest",
        "ureq",
        "isahc",
        "curl",
        "tauri-plugin-http",
        "tauri-plugin-shell",
        "tauri-plugin-updater",
        // telemetry and crash reporting
        "sentry",
        "tauri-plugin-sentry",
        "opentelemetry",
        "tauri-plugin-aptabase",
        "posthog-rs",
    ] {
        assert!(
            !packages.iter().any(|p| p == banned),
            "{target}: {banned} is in the build"
        );
    }
}

#[test]
fn the_rust_side_never_binds_a_socket() {
    for file in ["src/lib.rs", "src/main.rs"] {
        let src = read(file);
        for needle in ["TcpListener", "UdpSocket", "bind(", "std::net"] {
            assert!(!src.contains(needle), "{file} uses {needle}");
        }
    }
}

#[test]
fn transcript_text_reaches_the_page_only_as_text() {
    // model output derived from meeting audio is never parsed as HTML, in any window
    for file in ui_files(".js") {
        let js = read(&file);
        for needle in [
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "new Function",
        ] {
            assert!(!js.contains(needle), "{file} uses {needle}");
        }
    }
}

#[test]
fn every_window_is_one_of_the_apps_own_pages_under_the_same_permissions() {
    let caps: serde_json::Value = serde_json::from_str(&read("capabilities/default.json")).unwrap();
    let labels: Vec<&str> = caps["windows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap())
        .collect();
    let pages = ui_files(".html");
    for w in config()["app"]["windows"].as_array().unwrap() {
        let label = w["label"].as_str().unwrap();
        assert!(
            labels.contains(&label),
            "window {label} is outside the capability"
        );
        let url = w
            .get("url")
            .and_then(|u| u.as_str())
            .unwrap_or("index.html");
        assert!(
            pages.contains(&format!("../ui/{url}")),
            "window {label} loads {url}"
        );
    }
}
