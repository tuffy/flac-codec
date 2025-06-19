use flac_codec::metadata::{Cuesheet, Streaminfo, block};

fn main() {
    match std::env::args_os().skip(1).collect::<Vec<_>>().as_slice() {
        [flac, cuesheet] => match block::<_, Streaminfo>(&flac) {
            Ok(Some(Streaminfo {
                total_samples: Some(total_samples),
                ..
            })) => {
                match Cuesheet::parse(
                    total_samples.get(),
                    &std::fs::read_to_string(cuesheet).unwrap(),
                ) {
                    Ok(cuesheet) => {
                        dbg!(cuesheet);
                    }
                    Err(err) => eprintln!("* {}: {err}", cuesheet.display()),
                }
            }
            _ => eprintln!("* Unable to get total samples from FLAC file"),
        },
        _ => eprintln!("* Usage: flac-import-cuesheet <file.flac> <file.cue>"),
    }
}
