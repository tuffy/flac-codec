// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::decode::FlacSampleReader;
use flac_codec::encode::FlacSampleWriter;
use std::path::Path;

fn main() {
    // This is a CPU-heavy example which should be
    // using --release mode, or people might get confused
    // about how well it actually performs.
    if cfg!(debug_assertions) {
        eprintln!("WARNING: running in --release mode is preferred for best performance");
    }

    match std::env::args_os().skip(1).collect::<Vec<_>>().as_slice() {
        [first, rest @ .., last] => {
            if let Err(err) = concat_flacs(first, rest, last) {
                eprintln!("* Error: {err}");
            }
        }
        _ => eprintln!("* Usage: flac-cat <file 1.flac> [file 2.flac] ... <output.flac>"),
    }
}

fn concat_flacs<P: AsRef<Path>>(first: &P, rest: &[P], output: &P) -> Result<(), Error> {
    use flac_codec::{decode::Metadata, encode::Options};

    let mut first = FlacSampleReader::open(first)?;

    // ensure all files in the rest pile have consistent
    // parameters as those in the first
    // (a more full-featured concatenator could resample these if necessary,
    // but that's outside the scope of this crate)
    let rest = rest
        .iter()
        .map(|f| {
            FlacSampleReader::open(f)
                .map_err(Error::Flac)
                .and_then(|r| {
                    (r.sample_rate() == first.sample_rate()
                        && r.channel_count() == first.channel_count()
                        && r.bits_per_sample() == first.bits_per_sample())
                    .then_some(r)
                    .ok_or(Error::InconsistentParameters)
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut output = FlacSampleWriter::create(
        output,
        Options::default(),
        first.sample_rate(),
        first.bits_per_sample(),
        first.channel_count(),
        // if any input file's total samples are None, use None,
        // otherwise just sum them together
        // and multiply by channel count
        first.total_samples().and_then(|first_samples| {
            rest.iter()
                .map(|r| r.total_samples())
                .sum::<Option<u64>>()
                .map(|rest_samples| first_samples + rest_samples)
                .map(|total_samples| total_samples * u64::from(first.channel_count()))
        }),
    )?;

    // now do the real work to move samples from input files
    // to output file
    copy(&mut first, &mut output)?;

    for mut file in rest {
        copy(&mut file, &mut output)?;
    }

    output.finalize().map_err(Error::Flac)
}

fn copy<R, W>(
    r: &mut FlacSampleReader<R>,
    w: &mut FlacSampleWriter<W>,
) -> Result<(), flac_codec::Error>
where
    R: std::io::Read,
    W: std::io::Write + std::io::Seek,
{
    loop {
        match r.fill_buf()? {
            [] => break Ok(()),
            buf => {
                // borrow checker requires copy of buffer len
                let buf_len = buf.len();
                w.write(buf)?;
                r.consume(buf_len);
            }
        }
    }
}

#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    InconsistentParameters,
}

impl From<flac_codec::Error> for Error {
    fn from(err: flac_codec::Error) -> Self {
        Self::Flac(err)
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(flac) => flac.fmt(f),
            Self::InconsistentParameters => {
                "all files must have same samples rate, channels and bits-per-sample".fmt(f)
            }
        }
    }
}
