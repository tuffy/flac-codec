// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
