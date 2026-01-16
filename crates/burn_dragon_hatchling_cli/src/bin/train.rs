fn main() {
    #[cfg(feature = "cli")]
    {
        if let Err(err) = burn_dragon_hatchling::vision::train::run_cli() {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "cli"))]
    {
        eprintln!("train binary requires the `cli` feature");
        std::process::exit(1);
    }
}
