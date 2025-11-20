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
    use flac_codec::decode::{FlacChannelReader, Metadata};
    use flac_codec::encode::{FlacChannelWriter, Options};
    use rubato::{FftFixedInOut, Resampler};
    use std::collections::VecDeque;

    let mut decoder = FlacChannelReader::open(input.as_ref())?;

    let mut encoder = FlacChannelWriter::create(
        output.as_ref(),
        Options::default(),
        sample_rate as u32,
        decoder.bits_per_sample(),
        decoder.channel_count(),
        None, // won't know the exact number of output samples
    )?;

    let mut resampler: FftFixedInOut<f32> = FftFixedInOut::new(
        decoder.sample_rate() as usize,
        sample_rate,
        2,
        decoder.channel_count().into(),
    )?;

    let bps = decoder.metadata().streaminfo().bits_per_sample;
    let mut input_f: Vec<VecDeque<f32>> =
        vec![VecDeque::default(); usize::from(decoder.channel_count())];
    let mut output_f = resampler.output_buffer_allocate(true);
    let mut output_i: Vec<Vec<i32>> = vec![vec![]; usize::from(decoder.channel_count())];

    loop {
        match decoder.fill_buf()? {
            bufs if !bufs[0].is_empty() => {
                // entend floating point channel buffers with new samples
                for (i, b) in input_f.iter_mut().zip(bufs) {
                    i.extend(b.iter().copied().map(int_to_float(bps)));
                }
                decoder.consume(input_f[0].len());

                // process chunks of samples in input buffers
                while input_f[0].len() >= resampler.input_frames_next() {
                    let (input_frames, output_frames) = resampler.process_into_buffer(
                        &input_f
                            .iter_mut()
                            .map(|v| v.make_contiguous())
                            .collect::<Vec<_>>(),
                        &mut output_f,
                        None,
                    )?;

                    encoder.write(floats_to_ints(&output_f, &mut output_i, output_frames, bps))?;
                    input_f.iter_mut().for_each(|v| {
                        v.drain(0..input_frames);
                    });
                }
            }
            _ => {
                // process remaining chunks of samples in input buffers
                while input_f[0].len() >= resampler.input_frames_next() {
                    let (input_frames, output_frames) = resampler.process_into_buffer(
                        &input_f
                            .iter_mut()
                            .map(|v| v.make_contiguous())
                            .collect::<Vec<_>>(),
                        &mut output_f,
                        None,
                    )?;

                    encoder.write(floats_to_ints(&output_f, &mut output_i, output_frames, bps))?;
                    input_f.iter_mut().for_each(|v| {
                        v.drain(0..input_frames);
                    });
                }

                let output_frames = resampler
                    .process_partial_into_buffer(None::<&[&[f32]]>, &mut output_f, None)?
                    .1;
                encoder.write(floats_to_ints(&output_f, &mut output_i, output_frames, bps))?;
                break encoder.finalize().map_err(Error::Flac);
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

#[cfg(feature = "rubato")]
fn floats_to_ints<'i>(
    f: &[Vec<f32>],
    i: &'i mut Vec<Vec<i32>>,
    output_frames: usize,
    bps: SignedBitCount<32>,
) -> &'i [Vec<i32>] {
    for (f, i) in f.iter().zip(i.iter_mut()) {
        i.clear();
        i.extend(f.iter().copied().take(output_frames).map(float_to_int(bps)));
    }
    i.as_slice()
}

#[cfg(feature = "rubato")]
#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    RubatoConf(rubato::ResamplerConstructionError),
    Resample(rubato::ResampleError),
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
impl std::error::Error for Error {}

#[cfg(feature = "rubato")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(flac) => flac.fmt(f),
            Self::RubatoConf(r) => r.fmt(f),
            Self::Resample(r) => r.fmt(f),
        }
    }
}
