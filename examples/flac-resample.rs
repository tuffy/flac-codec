// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(feature = "rubato")]
use bitstream_io::SignedBitCount;

#[cfg(feature = "rubato")]
use std::path::Path;

#[cfg(feature = "rubato")]
fn main() {
    match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [input, output, sample_rate] => {
            if let Err(err) = resample(
                input,
                output,
                sample_rate.parse().expect("invalid sample rate integer"),
            ) {
                eprintln!("* {err}");
            }
        }
        _ => {
            eprintln!("* Usage: flac-resample <input.flac> <output.flac> <sample rate>");
        }
    }
}

#[cfg(not(feature = "rubato"))]
fn main() {
    eprintln!("* Enable the \"rubato\" feature to run this example");
}

#[cfg(feature = "rubato")]
fn resample<P: AsRef<Path>>(input: P, output: P, sample_rate: usize) -> Result<(), Error> {
    use flac_codec::decode::{FlacSampleReader, Metadata};
    use flac_codec::encode::{FlacSampleWriter, Options};
    use flac_codec::metadata::Streaminfo;
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Indexing, Resampler};

    const BUF_LEN: usize = 4096;

    let decoder = FlacSampleReader::open(input.as_ref())?;

    let mut encoder = FlacSampleWriter::create(
        output.as_ref(),
        Options::default(),
        sample_rate as u32,
        decoder.bits_per_sample(),
        decoder.channel_count(),
        None, // won't know the exact number of output samples
    )?;

    let mut resampler: Fft<f32> = Fft::new(
        decoder.sample_rate() as usize,
        sample_rate,
        1024,
        2,
        decoder.channel_count().into(),
        FixedSync::Both,
    )?;

    let Streaminfo {
        bits_per_sample: bps,
        channels,
        ..
    } = decoder.metadata().streaminfo();
    let bps = *bps;
    let channels: usize = channels.get().into();

    let mut decoder = FlacFloatSampleReader::new(decoder);

    let mut output_f = [0.0; BUF_LEN];
    let mut output_i = [0; BUF_LEN];

    let mut indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };

    loop {
        match decoder.fill_buf(resampler.input_frames_next() * channels)? {
            [] => {
                // finish encoding process
                break encoder.finalize().map_err(Error::Flac);
            }
            buf => {
                // should be None until the last iteration
                indexing.partial_len = (resampler.input_frames_next() > buf.len() / channels)
                    .then_some(buf.len() / channels);

                let (frames_read, frames_written) = resampler.process_into_buffer(
                    &InterleavedSlice::new(buf, channels, buf.len() / channels)?,
                    &mut InterleavedSlice::new_mut(&mut output_f, channels, BUF_LEN / channels)?,
                    Some(&indexing),
                )?;
                decoder.consume(frames_read * channels);

                let samples_written = frames_written * channels;

                // convert samples from floats back to ints
                for (i, o) in output_f[0..samples_written]
                    .iter()
                    .copied()
                    .map(float_to_int(bps))
                    .zip(&mut output_i[0..samples_written])
                {
                    *o = i;
                }

                // process int samples
                encoder.write(&output_i[0..samples_written])?;
            }
        }
    }
}

#[cfg(feature = "rubato")]
fn int_to_float(bps: SignedBitCount<32>) -> impl Fn(i32) -> f32 {
    // TODO - take advantage of unsigned_count once that stabilizes
    let shift = (1 << (u32::from(bps.count()) - 1)) as f32;

    move |i| i as f32 / shift
}

#[cfg(feature = "rubato")]
fn float_to_int(bps: SignedBitCount<32>) -> impl Fn(f32) -> i32 {
    // TODO - take advantage of unsigned_count once that stabilizes
    let shift = (1 << (u32::from(bps.count()) - 1)) as f32;
    let (min, max) = bps.range().into_inner();

    move |f| ((f * shift) as i32).clamp(min, max)
}

/// Reads FLAC samples as floating point values
#[cfg(feature = "rubato")]
struct FlacFloatSampleReader<R> {
    // the wrapped decoder
    decoder: flac_codec::decode::FlacSampleReader<R>,
    // decoded sample buffer
    buf: Vec<f32>,
}

#[cfg(feature = "rubato")]
impl<R: std::io::Read> FlacFloatSampleReader<R> {
    pub fn new(decoder: flac_codec::decode::FlacSampleReader<R>) -> Self {
        Self {
            decoder,
            buf: vec![],
        }
    }

    pub fn fill_buf(&mut self, required: usize) -> Result<&[f32], flac_codec::Error> {
        while self.buf.len() < required {
            let bps = self.decoder.metadata().streaminfo().bits_per_sample;
            match self.decoder.fill_buf()? {
                [] => break,
                buf => {
                    self.buf.extend(buf.iter().copied().map(int_to_float(bps)));
                    let buf_len = buf.len();
                    self.decoder.consume(buf_len);
                }
            }
        }

        Ok(self.buf.as_slice())
    }

    pub fn consume(&mut self, amt: usize) {
        let amt = amt.min(self.buf.len());
        self.buf.drain(0..amt);
    }
}

#[cfg(feature = "rubato")]
#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    RubatoConf(rubato::ResamplerConstructionError),
    Resample(rubato::ResampleError),
    Size(rubato::audioadapter_buffers::SizeError),
}

#[cfg(feature = "rubato")]
impl From<flac_codec::Error> for Error {
    fn from(err: flac_codec::Error) -> Self {
        Self::Flac(err)
    }
}

#[cfg(feature = "rubato")]
impl From<rubato::ResamplerConstructionError> for Error {
    fn from(err: rubato::ResamplerConstructionError) -> Self {
        Self::RubatoConf(err)
    }
}

#[cfg(feature = "rubato")]
impl From<rubato::ResampleError> for Error {
    fn from(err: rubato::ResampleError) -> Self {
        Self::Resample(err)
    }
}

#[cfg(feature = "rubato")]
impl From<rubato::audioadapter_buffers::SizeError> for Error {
    fn from(err: rubato::audioadapter_buffers::SizeError) -> Self {
        Self::Size(err)
    }
}

#[cfg(feature = "rubato")]
impl std::error::Error for Error {}

#[cfg(feature = "rubato")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(flac) => flac.fmt(f),
            Self::RubatoConf(r) => r.fmt(f),
            Self::Resample(r) => r.fmt(f),
            Self::Size(r) => r.fmt(f),
        }
    }
}
