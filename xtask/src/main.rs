use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about = "burn_dragon p2p task runner")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    BuildNative,
    BuildNativeWgpu,
    BuildNativeCuda,
    BuildBrowserCpu,
    BuildBrowser,
    BuildMatrix,
    NativeSmoke,
    NativeScale,
    NativeLarge,
    DowngradeSmoke,
    MixedFleet,
    EdgeDrill,
    WasmSmoke,
    CudaCheck,
    Smoke,
    DeployCheck,
    All,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::BuildNative => build_native(),
        CommandKind::BuildNativeWgpu => build_native_wgpu(),
        CommandKind::BuildNativeCuda => build_native_cuda(),
        CommandKind::BuildBrowserCpu => build_browser_cpu(),
        CommandKind::BuildBrowser => build_browser(),
        CommandKind::BuildMatrix => build_matrix(),
        CommandKind::NativeSmoke => native_smoke(),
        CommandKind::NativeScale => native_scale(),
        CommandKind::NativeLarge => native_large(),
        CommandKind::DowngradeSmoke => downgrade_smoke(),
        CommandKind::MixedFleet => mixed_fleet(),
        CommandKind::EdgeDrill => edge_drill(),
        CommandKind::WasmSmoke => wasm_smoke(),
        CommandKind::CudaCheck => cuda_check(),
        CommandKind::Smoke => smoke(),
        CommandKind::DeployCheck => deploy_check(),
        CommandKind::All => all(),
    }
}

const P2P_PACKAGE: &str = "burn_dragon_p2p";
const NATIVE_TEST: &str = "native_training";

#[derive(Clone, Copy)]
enum NativeBuildTarget {
    Cpu,
    Wgpu,
    Cuda,
}

impl NativeBuildTarget {
    fn features(self) -> &'static str {
        match self {
            Self::Cpu => "native",
            Self::Wgpu => "native,wgpu",
            Self::Cuda => "native,cuda",
        }
    }
}

#[derive(Clone, Copy)]
enum BrowserBuildTarget {
    Cpu,
    Wgpu,
}

impl BrowserBuildTarget {
    fn features(self) -> &'static str {
        match self {
            Self::Cpu => "wasm-ui,wasm-peer",
            Self::Wgpu => "wasm-ui,wasm-peer,wgpu",
        }
    }
}

fn smoke() -> Result<()> {
    native_smoke()?;
    wasm_smoke()?;
    build_native_cuda()?;
    Ok(())
}

fn all() -> Result<()> {
    build_matrix()?;
    deploy_check()
}

fn deploy_check() -> Result<()> {
    smoke()?;
    downgrade_smoke()?;
    native_scale()?;
    mixed_fleet()?;
    native_large()?;
    edge_drill()?;
    Ok(())
}

fn build_native() -> Result<()> {
    build_native_target(NativeBuildTarget::Cpu)
}

fn build_native_wgpu() -> Result<()> {
    build_native_target(NativeBuildTarget::Wgpu)
}

fn build_native_cuda() -> Result<()> {
    build_native_target(NativeBuildTarget::Cuda)
}

fn build_browser_cpu() -> Result<()> {
    build_browser_target(BrowserBuildTarget::Cpu)
}

fn build_browser() -> Result<()> {
    build_browser_target(BrowserBuildTarget::Wgpu)
}

fn build_matrix() -> Result<()> {
    build_native()?;
    build_native_wgpu()?;
    build_native_cuda()?;
    build_browser_cpu()?;
    build_browser()?;
    Ok(())
}

fn native_smoke() -> Result<()> {
    cargo_native_test(None, false)
}

fn native_scale() -> Result<()> {
    cargo_native_test(Some("medium_model"), true)
}

fn native_large() -> Result<()> {
    cargo_native_test(Some("large_model"), true)
}

fn mixed_fleet() -> Result<()> {
    cargo_native_test(Some("mixed_fleet"), false)?;
    cargo_native_test(Some("mixed_fleet"), true)
}

fn downgrade_smoke() -> Result<()> {
    cargo_native_test(Some("downgrades_to_validator"), false)?;
    wasm_smoke()
}

fn edge_drill() -> Result<()> {
    cargo_native_test(Some("edge_drill"), true)
}

fn wasm_smoke() -> Result<()> {
    let chrome = resolve_chrome_path()
        .context("could not find Google Chrome; install it or set BURN_DRAGON_PLAYWRIGHT_CHROME")?;
    let chromedriver = ensure_chromedriver(&chrome)?;
    let webdriver_json = write_webdriver_config(&chrome)?;
    let mut envs = vec![
        (
            OsString::from("CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER"),
            OsString::from("wasm-bindgen-test-runner"),
        ),
        (
            OsString::from("CHROMEDRIVER"),
            chromedriver.into_os_string(),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_ONLY_WEB"),
            OsString::from("1"),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_DRIVER_TIMEOUT"),
            OsString::from("60"),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_TIMEOUT"),
            OsString::from("180"),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_WEBDRIVER_JSON"),
            webdriver_json.into_os_string(),
        ),
    ];
    if let Some(existing) = std::env::var_os("PATH") {
        envs.push((OsString::from("PATH"), existing));
    }
    run_with_env(
        "cargo",
        &[
            "test",
            "-p",
            P2P_PACKAGE,
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
            "--features",
            BrowserBuildTarget::Wgpu.features(),
            "--lib",
            "--",
            "--nocapture",
        ],
        &envs,
    )
}

fn cuda_check() -> Result<()> {
    build_native_cuda()
}

fn cargo_native_test(filter: Option<&str>, ignored: bool) -> Result<()> {
    let mut args = vec![
        "test",
        "-p",
        P2P_PACKAGE,
        "--features",
        NativeBuildTarget::Wgpu.features(),
        "--test",
        NATIVE_TEST,
    ];
    if let Some(filter) = filter {
        args.push(filter);
    }
    args.push("--");
    if ignored {
        args.push("--ignored");
    }
    args.push("--nocapture");
    run("cargo", &args)
}

fn build_native_target(target: NativeBuildTarget) -> Result<()> {
    cargo_p2p_check(&["--features", target.features()])?;
    cargo_p2p_native_bin_check(target.features())
}

fn build_browser_target(target: BrowserBuildTarget) -> Result<()> {
    cargo_p2p_wasm_check(target.features())
}

fn cargo_p2p_check(extra_args: &[&str]) -> Result<()> {
    let mut args = vec!["check", "-p", P2P_PACKAGE];
    args.extend_from_slice(extra_args);
    run("cargo", &args)
}

fn cargo_p2p_native_bin_check(features: &str) -> Result<()> {
    run(
        "cargo",
        &[
            "check",
            "-p",
            P2P_PACKAGE,
            "--features",
            features,
            "--bin",
            "burn_dragon_p2p_native",
        ],
    )
}

fn cargo_p2p_wasm_check(features: &str) -> Result<()> {
    run(
        "cargo",
        &[
            "check",
            "-p",
            P2P_PACKAGE,
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
            "--features",
            features,
        ],
    )
}

fn run(program: &str, args: &[&str]) -> Result<()> {
    run_with_env(program, args, &[])
}

fn run_with_env(program: &str, args: &[&str], envs: &[(OsString, OsString)]) -> Result<()> {
    eprintln!("+ {} {}", program, args.join(" "));
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::null());
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command
        .status()
        .with_context(|| format!("failed to start `{program}`"))?;
    if !status.success() {
        bail!("command failed: {} {}", program, args.join(" "));
    }
    Ok(())
}

fn ensure_chromedriver(chrome: &Path) -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("CHROMEDRIVER")
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        return Ok(explicit);
    }
    if let Some(found) = find_in_path("chromedriver") {
        return Ok(found);
    }

    let version = chrome_version(chrome)?;
    let cache_root = workspace_root()
        .join("target")
        .join("xtask")
        .join("chromedriver")
        .join(&version);
    let binary = cache_root.join("chromedriver-linux64").join("chromedriver");
    if binary.is_file() {
        return Ok(binary);
    }

    fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "failed to create chromedriver cache {}",
            cache_root.display()
        )
    })?;
    let zip_path = cache_root.join("chromedriver-linux64.zip");
    let url = format!(
        "https://storage.googleapis.com/chrome-for-testing-public/{version}/linux64/chromedriver-linux64.zip"
    );

    run(
        "curl",
        &[
            "-fsSL",
            &url,
            "-o",
            zip_path.to_str().context("zip path was not valid utf-8")?,
        ],
    )
    .with_context(|| format!("failed to download chromedriver for Chrome {version}"))?;
    run(
        "unzip",
        &[
            "-oq",
            zip_path.to_str().context("zip path was not valid utf-8")?,
            "-d",
            cache_root
                .to_str()
                .context("cache root path was not valid utf-8")?,
        ],
    )
    .context("failed to unzip chromedriver bundle")?;

    if !binary.is_file() {
        bail!(
            "downloaded chromedriver bundle did not contain expected binary at {}",
            binary.display()
        );
    }
    Ok(binary)
}

fn chrome_version(chrome: &Path) -> Result<String> {
    let output = Command::new(chrome)
        .arg("--version")
        .output()
        .with_context(|| format!("failed to run `{}` --version", chrome.display()))?;
    if !output.status.success() {
        bail!("`{}` --version failed", chrome.display());
    }
    let stdout = String::from_utf8(output.stdout).context("chrome version output was not utf-8")?;
    stdout
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
        .map(str::to_owned)
        .context("failed to parse Chrome version")
}

fn write_webdriver_config(chrome: &Path) -> Result<PathBuf> {
    let config_path = workspace_root()
        .join("target")
        .join("xtask")
        .join("dragon-webdriver.json");
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create webdriver config directory {}",
                parent.display()
            )
        })?;
    }
    let payload = format!(
        concat!(
            "{{\n",
            "  \"goog:chromeOptions\": {{\n",
            "    \"binary\": \"{}\",\n",
            "    \"args\": [\n",
            "      \"--enable-unsafe-webgpu\",\n",
            "      \"--use-angle=swiftshader\"\n",
            "    ]\n",
            "  }}\n",
            "}}\n"
        ),
        chrome.display()
    );
    fs::write(&config_path, payload).with_context(|| {
        format!(
            "failed to write webdriver config to {}",
            config_path.display()
        )
    })?;
    Ok(config_path)
}

fn resolve_chrome_path() -> Option<PathBuf> {
    resolve_binary(
        "BURN_DRAGON_PLAYWRIGHT_CHROME",
        &[
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ],
    )
    .or_else(|| {
        resolve_binary(
            "BURN_P2P_PLAYWRIGHT_CHROME",
            &[
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ],
        )
    })
}

fn resolve_binary(env_var: &str, candidates: &[&str]) -> Option<PathBuf> {
    std::env::var_os(env_var)
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            candidates
                .iter()
                .map(PathBuf::from)
                .find(|path| path.exists())
        })
}

fn find_in_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live under workspace root")
        .to_path_buf()
}
