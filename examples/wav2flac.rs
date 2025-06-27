// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ffi::OsString;
use std::path::Path;

fn main() {
    match std::env::args_os().skip(1).collect::<Vec<_>>().as_slice() {
        [] => eprintln!("* Usage: wav2flac [file 1.wav] [file 2.wav] ..."),
        flacs => {
            if let Err(err) = wav2flac(flacs) {
                eprintln!("* Error: {err}");
            }
        }
    }
}

fn wav2flac(wavs: &[OsString]) -> Result<(), Error> {
    for wav in wavs {
        convert_wav(wav.as_ref())?;
    }
    Ok(())
}

fn convert_wav(wav: &Path) -> Result<(), Error> {
    use bitstream_io::{ByteRead, ByteReader};
    use flac_codec::byteorder::LittleEndian;
    use flac_codec::encode::{FlacByteWriter, Options};
    use std::fs::File;
    use std::io::{BufReader, Read};

    let flac_path = wav.with_extension("flac");
    if flac_path.exists() {
        eprintln!("{} already exists, skipping...", flac_path.display());
        return Ok(());
    }

    let mut wav = ByteReader::endian(BufReader::new(File::open(wav)?), bitstream_io::LittleEndian);

    // "RIFF"
    if wav.parse::<Header>()?.id != [0x52, 0x49, 0x46, 0x46] {
        return Err(Error::InvalidWave);
    }
    // "WAVE"
    if wav.read::<[u8; 4]>()? != [0x57, 0x41, 0x56, 0x45] {
        return Err(Error::InvalidWave);
    }

    let mut fmt = None;

    loop {
        match wav.parse::<Header>()? {
            // "fmt " chunk
            Header {
                id: [0x66, 0x6d, 0x74, 0x20],
                ..
            } => {
                fmt = Some(wav.parse::<Fmt>()?);
            }
            // "data" chunk
            Header {
                id: [0x64, 0x61, 0x74, 0x61],
                size,
            } => {
                // "data" chunk must come after "fmt " chunk
                let fmt = fmt.ok_or(Error::InvalidWave)?;

                let mut flac: FlacByteWriter<_, LittleEndian> = FlacByteWriter::create(
                    &flac_path,
                    match fmt.channel_mask() {
                        None => Options::default(),
                        Some(channel_mask) => {
                            use flac_codec::metadata::{ChannelMask, fields::CHANNEL_MASK};

                            Options::default().tag(CHANNEL_MASK, ChannelMask::from(channel_mask))
                        }
                    },
                    fmt.sample_rate(),
                    fmt.bits_per_sample().into(),
                    fmt.channels()
                        .try_into()
                        .map_err(|_| Error::ExcessiveChannels)?,
                    Some(size.into()),
                )?;

                std::io::copy(&mut wav.reader().take(size.into()), &mut flac)?;

                break flac
                    .finalize()
                    .inspect(|_| {
                        println!("* Wrote: {}", flac_path.display());
                    })
                    .map_err(Error::Flac);
            }
            // skip other chunks
            Header { size, .. } => {
                // chunks must end on even byte boundaries
                wav.skip(size + if size % 2 == 1 { 1 } else { 0 })?;
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Header {
    id: [u8; 4],
    size: u32,
}

impl bitstream_io::FromByteStream for Header {
    type Error = std::io::Error;

    fn from_reader<R>(r: &mut R) -> Result<Self, Self::Error>
    where
        R: bitstream_io::ByteRead + ?Sized,
    {
        Ok(Self {
            id: r.read()?,
            size: r.read()?,
        })
    }
}

#[derive(Clone, Debug)]
enum Fmt {
    Standard {
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
    },
    Extensible {
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        channel_mask: u32,
    },
}

impl Fmt {
    fn channels(&self) -> u16 {
        match self {
            Self::Standard { channels, .. } | Self::Extensible { channels, .. } => *channels,
        }
    }

    fn sample_rate(&self) -> u32 {
        match self {
            Self::Standard { sample_rate, .. } | Self::Extensible { sample_rate, .. } => {
                *sample_rate
            }
        }
    }

    fn bits_per_sample(&self) -> u16 {
        match self {
            Self::Standard {
                bits_per_sample, ..
            }
            | Self::Extensible {
                bits_per_sample, ..
            } => *bits_per_sample,
        }
    }

    fn channel_mask(&self) -> Option<u32> {
        match self {
            Self::Standard { .. } => None,
            Self::Extensible { channel_mask, .. } => Some(*channel_mask),
        }
    }
}

impl bitstream_io::FromByteStream for Fmt {
    type Error = Error;

    fn from_reader<R>(r: &mut R) -> Result<Self, Self::Error>
    where
        R: bitstream_io::ByteRead + ?Sized,
    {
        match r.read::<u16>()? {
            0x0001 => {
                let channels = r.read()?;
                let sample_rate = r.read()?;
                let _data_rate = r.read::<u32>()?;
                let _data_block_size = r.read::<u16>()?;
                Ok(Self::Standard {
                    channels,
                    sample_rate,
                    bits_per_sample: r.read()?,
                })
            }
            0xFFFE => {
                let channels = r.read()?;
                let sample_rate = r.read()?;
                let _data_rate = r.read::<u32>()?;
                let _data_block_size = r.read::<u16>()?;
                let bits_per_sample = r.read()?;

                // extension size should be 22 bytes
                if r.read::<u16>()? != 22 {
                    return Err(Error::InvalidWave);
                }

                let _valid_bits = r.read::<u16>()?;
                let channel_mask = r.read()?;
                let _sub_format = r.read::<[u8; 16]>()?;

                Ok(Self::Extensible {
                    channels,
                    sample_rate,
                    bits_per_sample,
                    channel_mask,
                })
            }
            _ => Err(Error::UnsupportedFormat),
        }
    }
}

#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    InvalidWave,
    UnsupportedFormat,
    ExcessiveChannels,
}

impl From<flac_codec::Error> for Error {
    fn from(err: flac_codec::Error) -> Error {
        Error::Flac(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::Flac(flac_codec::Error::Io(err))
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(err) => err.fmt(f),
            Self::InvalidWave => "invalid RIFF WAVE file".fmt(f),
            Self::UnsupportedFormat => "unsupported fmt chunk".fmt(f),
            Self::ExcessiveChannels => "too many channels for FLAC file".fmt(f),
        }
    }
}
