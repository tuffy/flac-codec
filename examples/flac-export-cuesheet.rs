// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::{Cuesheet, block};

/// Roughly corresponds to the reference implementation's:
///
/// "metaflac --export-cuesheet-to=<file.cue> <file.flac>"
///
/// Except we simply dump the cuesheet to stdout instead
/// of to a new file.

fn main() {
    match std::env::args().skip(1).next() {
        Some(flac) => match block::<_, Cuesheet>(&flac) {
            Ok(Some(cuesheet)) => print!("{}", cuesheet.display(&flac)),
            Ok(None) => eprintln!("no embedded cuesheet found"),
            Err(err) => eprintln!("* {flac}: {err}"),
        },
        None => eprintln!("* Usage: flac-export-cuesheet <file.flac>"),
    }
}
