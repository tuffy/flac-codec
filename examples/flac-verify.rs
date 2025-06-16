use flac_codec::{
    Error,
    decode::{Verified, verify},
};
use std::path::PathBuf;

/// This correponds to the reference implementation's:
///
/// "flac -t <file1.flac> <file2.flac> ..."
///
/// In that it verifies FLAC files for correctness
/// and displays the results.
///
/// Runs in parallel with the "rayon" feature enabled.

fn main() {
    // This is a CPU-heavy example which should be
    // using --release mode, or people might get confused
    // about how well it actually performs.
    if cfg!(debug_assertions) {
        eprintln!("WARNING: running in --release mode is preferred for best performance");
    }

    for (result, file) in verify_inputs(std::env::args_os().skip(1).map(PathBuf::from).collect()) {
        display_result(file, result);
    }
}

#[cfg(not(feature = "rayon"))]
fn verify_inputs(inputs: Vec<PathBuf>) -> Vec<(Result<Verified, Error>, PathBuf)> {
    inputs.into_iter().map(|pb| (verify(&pb), pb)).collect()
}

#[cfg(feature = "rayon")]
fn verify_inputs(inputs: Vec<PathBuf>) -> Vec<(Result<Verified, Error>, PathBuf)> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    inputs.into_par_iter().map(|pb| (verify(&pb), pb)).collect()
}

fn display_result(file: PathBuf, result: Result<Verified, Error>) {
    match result {
        Ok(Verified::MD5Match) => println!("{}: ok", file.display()),
        Ok(Verified::MD5Mismatch) => println!("{}: bad - MD5 mismatch", file.display()),
        Ok(Verified::NoMD5) => println!("{}: ok - no MD5", file.display()),
        Err(err) => println!("{}: error - {}", file.display(), err),
    }
}
