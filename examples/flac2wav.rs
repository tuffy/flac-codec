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
        [] => eprintln!("* Usage: flac2wav [file 1.flac] [file 2.flac] ..."),
        flacs => {
            if let Err(err) = flac2wav(flacs) {
                eprintln!("* Error: {err}");
            }
        }
    }
}

#[cfg(not(feature = "rayon"))]
fn flac2wav(flacs: &[OsString]) -> Result<(), Error> {
    for flac in flacs {
        convert_flac(flac.as_ref())?;
    }
    Ok(())
}

#[cfg(feature = "rayon")]
fn flac2wav(flacs: &[OsString]) -> Result<(), Error> {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    flacs
        .par_iter()
        .try_for_each(|flac| convert_flac(flac.as_ref()))
}

fn convert_flac(flac: &Path) -> Result<(), Error> {
    use flac_codec::byteorder::LittleEndian;
    use flac_codec::decode::FlacReader;
    use flac_codec::metadata::Application;
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let wav_path = flac.with_extension("wav");
    if wav_path.exists() {
        eprintln!("{} already exists, skipping...", wav_path.display());
        return Ok(());
    }

    let mut flac = FlacReader::open(flac, LittleEndian)?;
    let mut wav = File::create_new(&wav_path).map(BufWriter::new)?;

    match flac
        .metadata()
        .get_all::<Application>()
        .filter(|a| a.id == Application::RIFF)
        .cloned()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => {
            use bitstream_io::{ByteWrite, ByteWriter, LittleEndian, ToByteStream};
            use flac_codec::metadata::Metadata;

            // FLAC file has no foreign RIFF chunks,
            // so construct RIFF WAVE chunks if possible
            let fmt = Fmt::new(&flac);

            let data_size: u32 = flac
                .decoded_len()
                .ok_or(Error::FlacSizeUnknown)?
                .try_into()
                .map_err(|_| Error::FlacTooLarge)?;

            let whole_size: u32 = (4 + fmt.bytes::<u32>()? + 8)
                .checked_add(data_size)
                .ok_or(Error::FlacTooLarge)?;

            let mut wav = ByteWriter::endian(wav, LittleEndian);
            wav.write_bytes(b"RIFF")?;
            wav.write(whole_size)?;
            wav.write_bytes(b"WAVE")?;
            wav.build(&fmt)?;
            wav.write_bytes(b"data")?;
            wav.write(data_size)?;
            std::io::copy(&mut flac, wav.writer())?;
            println!("* Write: {}", wav_path.display());
            Ok(())
        }
        chunks => {
            // FLAC file has foreign RIFF chunks,
            // so preserve them in output file
            for chunk in chunks {
                match chunk.data.as_slice() {
                    chunk @ &[0x64, 0x61, 0x74, 0x61, _, _, _, _] => {
                        // data chunk needs bytes populated from reader
                        wav.write_all(&chunk)?;
                        std::io::copy(&mut flac, &mut wav)?;
                    }
                    chunk => {
                        // other chunks output verbatim
                        wav.write_all(&chunk)?;
                    }
                }
            }
            println!("* Wrote: {}", wav_path.display());
            Ok(())
        }
    }
}

enum Fmt {
    Standard {
        channels: u16,
        sample_rate: u32,
        data_rate: u32,
        data_block_size: u16,
        bits_per_sample: u16,
    },
    Extensible {
        channels: u16,
        sample_rate: u32,
        data_rate: u32,
        data_block_size: u16,
        bits_per_sample: u16,
        valid_bits: u16,
        channel_mask: u32,
        sub_format: [u8; 16],
    },
}

impl Fmt {
    fn new<M>(m: &M) -> Self
    where
        M: flac_codec::metadata::Metadata,
    {
        if m.channel_count() <= 2 && m.bits_per_sample() <= 16 {
            Self::Standard {
                channels: m.channel_count().into(),
                sample_rate: m.sample_rate(),
                data_rate: m.sample_rate()
                    * (m.bits_per_sample() / 8)
                    * u32::from(m.channel_count()),
                data_block_size: u16::try_from(m.bits_per_sample() / 8).unwrap()
                    * u16::from(m.channel_count()),
                bits_per_sample: m.bits_per_sample().try_into().unwrap(),
            }
        } else {
            Self::Extensible {
                channels: m.channel_count().into(),
                sample_rate: m.sample_rate(),
                data_rate: m.sample_rate()
                    * (m.bits_per_sample() / 8)
                    * u32::from(m.channel_count()),
                data_block_size: u16::try_from(m.bits_per_sample() / 8).unwrap()
                    * u16::from(m.channel_count()),
                bits_per_sample: m.bits_per_sample().try_into().unwrap(),
                valid_bits: m.bits_per_sample().try_into().unwrap(),
                channel_mask: m.channel_mask().into(),
                sub_format: [
                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00,
                    0x38, 0x9B, 0x71,
                ],
            }
        }
    }
}

impl bitstream_io::ToByteStream for Fmt {
    type Error = std::io::Error;

    // yields entire fmt chunk, header included
    fn to_writer<W>(&self, w: &mut W) -> std::io::Result<()>
    where
        W: bitstream_io::ByteWrite + ?Sized,
    {
        match self {
            Self::Standard {
                channels,
                sample_rate,
                data_rate,
                data_block_size,
                bits_per_sample,
            } => {
                w.write_bytes(b"fmt ")?; // chunk ID
                w.write::<u32>(16)?; // chunk size
                w.write::<u16>(0x0001)?; // WAVE_FORMAT_PCM
                w.write(*channels)?;
                w.write(*sample_rate)?;
                w.write(*data_rate)?;
                w.write(*data_block_size)?;
                w.write(*bits_per_sample)?;
                Ok(())
            }
            Self::Extensible {
                channels,
                sample_rate,
                data_rate,
                data_block_size,
                bits_per_sample,
                valid_bits,
                channel_mask,
                sub_format,
            } => {
                w.write_bytes(b"fmt ")?; // chunk ID
                w.write::<u32>(40)?; // chunk size
                w.write::<u16>(0xFFFE)?; // WAVE_FORMAT_EXTENSIBLE
                w.write(*channels)?;
                w.write(*sample_rate)?;
                w.write(*data_rate)?;
                w.write(*data_block_size)?;
                w.write(*bits_per_sample)?;
                w.write::<u16>(22)?; // size of extension
                w.write(*valid_bits)?;
                w.write(*channel_mask)?;
                w.write_bytes(sub_format)?;
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
enum Error {
    FlacTooLarge,
    FlacSizeUnknown,
    Flac(flac_codec::Error),
}

impl From<flac_codec::Error> for Error {
    fn from(error: flac_codec::Error) -> Self {
        Self::Flac(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Flac(flac_codec::Error::Io(error))
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::FlacTooLarge => "FLAC file too large for .wav file".fmt(f),
            Self::FlacSizeUnknown => "Decoded FLAC file size is unknown".fmt(f),
            Self::Flac(err) => err.fmt(f),
        }
    }
}
