// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::{Cuesheet, Metadata, update};

fn main() {
    match std::env::args_os().skip(1).collect::<Vec<_>>().as_slice() {
        [flac, cuesheet] => {
            if let Err(err) = update(flac, |blocks| {
                if blocks.has::<Cuesheet>() {
                    Err(Error::DuplicateCuesheet)
                } else {
                    blocks.insert(Cuesheet::parse(
                        blocks.total_samples().ok_or(Error::NoTotalSamples)?,
                        &std::fs::read_to_string(cuesheet)?,
                    )?);
                    Ok(())
                }
            }) {
                eprintln!("* {err}")
            }
        }
        _ => eprintln!("* Usage: flac-import-cuesheet <file.flac> <file.cue>"),
    }
}

#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    DuplicateCuesheet,
    NoTotalSamples,
}

impl From<flac_codec::Error> for Error {
    fn from(err: flac_codec::Error) -> Self {
        Self::Flac(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Flac(flac_codec::Error::Io(err))
    }
}

impl From<flac_codec::metadata::CuesheetError> for Error {
    fn from(err: flac_codec::metadata::CuesheetError) -> Self {
        Self::Flac(flac_codec::Error::Cuesheet(err))
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(flac) => flac.fmt(f),
            Self::DuplicateCuesheet => "FLAC file already has CUESHEET block".fmt(f),
            Self::NoTotalSamples => "unknown total samples in FLAC file".fmt(f),
        }
    }
}
