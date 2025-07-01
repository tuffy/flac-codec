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

    match std::env::args_os().nth(1) {
        Some(source) => {
            if let Err(err) = split_flac(source) {
                eprintln!("* {err}");
            }
        }
        None => {
            eprintln!("* Usage: flac-split <source.flac>");
        }
    }
}

fn split_flac<P: AsRef<Path>>(source: P) -> Result<(), Error> {
    use flac_codec::metadata::{Cuesheet, block};
    use std::io::Cursor;

    if block::<_, Cuesheet>(source.as_ref())?.is_none() {
        return Err(Error::NoCuesheet);
    }

    // read whole source FLAC file at once
    // and access its contents through a cursor
    // (this may come in handy later)
    let source_data = std::fs::read(source.as_ref())?;

    let source_reader = FlacSampleReader::new_seekable(Cursor::new(source_data.as_slice()))?;

    let track_ranges = source_reader
        .metadata()
        .get::<Cuesheet>()
        .expect("CUESHEET not found")
        .track_sample_ranges()
        .zip(1..)
        .collect::<Vec<_>>();

    extract_tracks(track_ranges, source.as_ref(), source_reader)
}

#[cfg(not(feature = "rayon"))]
fn extract_tracks<R>(
    track_ranges: Vec<(std::ops::Range<u64>, u8)>,
    source: &Path,
    mut reader: FlacSampleReader<R>,
) -> Result<(), Error>
where
    R: std::io::Read + std::io::Seek + Clone + Sync,
{
    track_ranges
        .into_iter()
        .try_for_each(|(sample_range, track_num)| {
            extract_track(source.as_ref(), track_num, &mut reader, sample_range)
        })
}

#[cfg(feature = "rayon")]
fn extract_tracks<R>(
    track_ranges: Vec<(std::ops::Range<u64>, u8)>,
    source: &Path,
    reader: FlacSampleReader<R>,
) -> Result<(), Error>
where
    R: std::io::Read + std::io::Seek + Clone + Sync,
{
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    track_ranges
        .into_par_iter()
        .try_for_each(|(sample_range, track_num)| {
            extract_track(
                source.as_ref(),
                track_num,
                &mut reader.clone(),
                sample_range,
            )
        })
}

fn extract_track<R: std::io::Read + std::io::Seek>(
    source: &Path,
    track_num: u8,
    reader: &mut FlacSampleReader<R>,
    std::ops::Range { start, end }: std::ops::Range<u64>,
) -> Result<(), Error> {
    use flac_codec::{decode::Metadata, encode::Options};

    match source.file_stem() {
        Some(file_stem) => {
            let total_samples = (end - start) * u64::from(reader.channel_count());

            let mut split_name = file_stem.to_os_string();
            split_name.push(format!("-track{:02}.flac", track_num));

            reader.seek(start)?;

            let mut writer = FlacSampleWriter::create(
                &split_name,
                Options::default(),
                reader.sample_rate(),
                reader.bits_per_sample(),
                reader.channel_count(),
                Some(total_samples),
            )?;

            copy(reader, &mut writer, total_samples)?;

            writer.finalize().map_err(Error::Flac).inspect(|()| {
                println!("* Wrote : {}", split_name.display());
            })
        }
        // no file name, so simply skip file
        None => Ok(()),
    }
}

fn copy<R, W>(
    reader: &mut FlacSampleReader<R>,
    writer: &mut FlacSampleWriter<W>,
    mut samples: u64,
) -> Result<(), flac_codec::Error>
where
    R: std::io::Read,
    W: std::io::Write + std::io::Seek,
{
    while samples > 0 {
        match reader.fill_buf()? {
            [] => return Ok(()),
            buf => {
                let to_write = usize::try_from(samples)
                    .map(|s| buf.len().min(s))
                    .unwrap_or(buf.len());
                writer.write(&buf[0..to_write])?;
                reader.consume(to_write);
                samples -= to_write as u64;
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    NoCuesheet,
}

impl From<flac_codec::Error> for Error {
    fn from(err: flac_codec::Error) -> Error {
        Error::Flac(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Flac(flac_codec::Error::Io(err))
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(err) => err.fmt(f),
            Self::NoCuesheet => "no CUESHEET block in source file".fmt(f),
        }
    }
}
