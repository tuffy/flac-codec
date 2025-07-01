// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(feature = "cpal")]
use flac_codec::decode::FlacSampleReader;
#[cfg(feature = "cpal")]
use std::path::Path;

/// Plays FLAC files to whatever default output device it can find

#[cfg(feature = "cpal")]
fn main() {
    use cpal::traits::HostTrait;

    // This is a CPU-heavy example which should be
    // using --release mode, or people might get confused
    // about how well it actually performs.
    if cfg!(debug_assertions) {
        eprintln!("WARNING: running in --release mode is preferred for best performance");
    }

    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .expect("failed to find output device");

    for flac in std::env::args_os().skip(1) {
        if let Err(err) = play_flac(&flac, &device) {
            eprintln!("* {}: {err}", flac.display());
        }
    }
}


#[cfg(not(feature = "cpal"))]
fn main() {
    eprintln!("* Enable the \"cpal\" feature to run this example");
}

#[cfg(feature = "cpal")]
fn play_flac<P: AsRef<Path>>(flac: P, device: &cpal::Device) -> Result<(), Error> {
    use cpal::traits::{DeviceTrait, StreamTrait};
    use flac_codec::decode::Metadata;

    let mut flac = FlacSampleReader::open(flac.as_ref())?;
    let flac_duration = flac.duration().expect("FLAC duration must be known");

    let stream = device.build_output_stream(
        &cpal::StreamConfig {
            channels: flac.channel_count().into(),
            sample_rate: cpal::SampleRate(flac.sample_rate()),
            buffer_size: cpal::BufferSize::Default,
        },
        move |buf: &mut [f32], _| output_flac_data(&mut flac, buf),
        |err| eprintln!("* {err}"),
        None,
    )?;
    stream.play()?;
    std::thread::sleep(flac_duration);
    Ok(())
}

#[cfg(feature = "cpal")]
fn output_flac_data<R>(flac: &mut FlacSampleReader<R>, mut output_buf: &mut [f32])
where
    R: std::io::Read,
{
    use flac_codec::decode::Metadata;

    let shift = (1 << flac.bits_per_sample() - 1) as f32;

    while !output_buf.is_empty() {
        match flac.fill_buf() {
            Ok([]) => {
                // FLAC file is exhausted, but output buf wants more data
                // so pad out with silence
                output_buf.fill(0.0);
                return;
            }
            Ok(flac_buf) => {
                let to_consume = samples_to_floats(flac_buf, output_buf, shift);
                output_buf = &mut output_buf[to_consume..];
                flac.consume(to_consume);
            }
            Err(err) => {
                // nothing to do but simply display an error
                // if one should happen to occur
                eprintln!("* {err}");
                output_buf.fill(0.0);
                return;
            }
        }
    }
}

#[cfg(feature = "cpal")]
fn samples_to_floats(flac: &[i32], output: &mut [f32], shift: f32) -> usize {
    let mut count = 0;
    for (i, o) in flac.iter().zip(output.iter_mut()) {
        *o = *i as f32 / shift;
        count += 1;
    }
    count
}

#[cfg(feature = "cpal")]
#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    BuildStream(cpal::BuildStreamError),
    PlayStream(cpal::PlayStreamError),
}

#[cfg(feature = "cpal")]
impl std::error::Error for Error {}

#[cfg(feature = "cpal")]
impl From<flac_codec::Error> for Error {
    fn from(err: flac_codec::Error) -> Self {
        Self::Flac(err)
    }
}

#[cfg(feature = "cpal")]
impl From<cpal::BuildStreamError> for Error {
    fn from(err: cpal::BuildStreamError) -> Self {
        Self::BuildStream(err)
    }
}

#[cfg(feature = "cpal")]
impl From<cpal::PlayStreamError> for Error {
    fn from(err: cpal::PlayStreamError) -> Self {
        Self::PlayStream(err)
    }
}

#[cfg(feature = "cpal")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(flac) => flac.fmt(f),
            Self::BuildStream(stream) => stream.fmt(f),
            Self::PlayStream(stream) => stream.fmt(f),
        }
    }
}
