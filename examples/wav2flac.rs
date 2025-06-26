// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::Application;
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
    use bitstream_io::{ByteRead, ByteReader, LittleEndian};
    use flac_codec::encode::{FlacWriter, Options};
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};

    let flac_path = wav.with_extension("flac");
    if flac_path.exists() {
        eprintln!("{} already exists, skipping...", flac_path.display());
        return Ok(());
    }

    let mut wav = ByteReader::endian(BufReader::new(File::open(wav)?), LittleEndian);
    let mut fmt_chunk = None;
    let mut data_offset = None;
    let mut data_size = None;

    let file_header = wav.parse::<FileHeader>()?;

    let mut chunks = vec![Chunk::Header(file_header)];

    let mut remaining_bytes = file_header
        .file_size
        .checked_sub(4)
        .ok_or(Error::InvalidWave)?;

    while remaining_bytes > 0 {
        let chunk_header = wav.parse::<Header>()?;
        remaining_bytes = remaining_bytes.checked_sub(8).ok_or(Error::InvalidWave)?;

        match chunk_header.id {
            // "fmt " chunk
            [0x66, 0x6d, 0x74, 0x20] => {
                if fmt_chunk.is_some() {
                    // multiple "fmt " chunks is invalid
                    return Err(Error::InvalidWave);
                } else {
                    let fmt = wav.parse::<Fmt>()?;
                    chunks.push(Chunk::Fmt(fmt.clone()));
                    fmt_chunk = Some(fmt);
                }
            }
            // "data" chunk
            [0x64, 0x61, 0x74, 0x61] => {
                if fmt_chunk.is_none() || data_offset.is_some() {
                    // multiple "data" chunks is invalid
                    return Err(Error::InvalidWave);
                } else {
                    chunks.push(Chunk::Data {
                        header: chunk_header,
                    });

                    data_offset = Some(wav.reader().stream_position()?);

                    data_size = Some(chunk_header.size);
                    // the data chunk could potentially be very large,
                    // so seek over it instead of using a skip
                    wav.reader()
                        .seek(SeekFrom::Current(chunk_header.size.into()))?;
                }
            }
            // other foreign chunks
            _ => {
                let mut chunk_data = vec![0; chunk_header.size.try_into().unwrap()];
                wav.read_bytes(&mut chunk_data)?;
                chunks.push(Chunk::Foreign {
                    header: chunk_header,
                    data: chunk_data,
                });
            }
        }

        remaining_bytes = remaining_bytes
            .checked_sub(chunk_header.size)
            .ok_or(Error::InvalidWave)?;

        if chunk_header.size % 2 == 1 {
            wav.skip(1)?;
            remaining_bytes = remaining_bytes.checked_sub(1).ok_or(Error::InvalidWave)?;
        }
    }

    // we've gone through all the chunks, now to write the FLAC itself
    // (which is relatively easy, by comparison)

    // these must be present in a valid RIFF WAVE file
    let fmt_chunk = fmt_chunk.ok_or(Error::InvalidWave)?;
    let data_offset = data_offset.ok_or(Error::InvalidWave)?;
    let data_size = data_size.ok_or(Error::InvalidWave)?;

    if !chunks.iter().any(|c| c.is_foreign()) {
        // write a plain FLAC file
        let mut flac: FlacWriter<_, flac_codec::byteorder::LittleEndian> = FlacWriter::create(
            &flac_path,
            Options::default(),
            fmt_chunk.sample_rate(),
            fmt_chunk.bits_per_sample().into(),
            fmt_chunk
                .channels()
                .try_into()
                .map_err(|_| Error::ExcessiveChannels)?,
            Some(data_size.into()),
        )?;

        wav.reader().seek(SeekFrom::Start(data_offset))?;

        std::io::copy(wav.reader(), &mut flac)?;

        flac.finalize().map_err(Error::Flac).inspect(|_| {
            println!("* Wrote: {}", flac_path.display());
        })
    } else {
        // write a FLAC file populated with foreign chunks
        let mut options = Options::default();

        for chunk in chunks {
            options = options.application(chunk.into());
        }

        let mut flac: FlacWriter<_, flac_codec::byteorder::LittleEndian> = FlacWriter::create(
            &flac_path,
            options,
            fmt_chunk.sample_rate(),
            fmt_chunk.bits_per_sample().into(),
            fmt_chunk
                .channels()
                .try_into()
                .map_err(|_| Error::ExcessiveChannels)?,
            Some(data_size.into()),
        )?;

        wav.reader().seek(SeekFrom::Start(data_offset))?;

        std::io::copy(&mut wav.reader().take(data_size.into()), &mut flac)?;

        flac.finalize().map_err(Error::Flac).inspect(|_| {
            println!("* Wrote: {}", flac_path.display());
        })
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

impl bitstream_io::ToByteStream for Header {
    type Error = std::io::Error;

    fn to_writer<W>(&self, w: &mut W) -> Result<(), Self::Error>
    where
        W: bitstream_io::ByteWrite + ?Sized,
    {
        w.write(self.id)?;
        w.write(self.size)?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
struct FileHeader {
    file_size: u32,
}

impl bitstream_io::FromByteStream for FileHeader {
    type Error = Error;

    fn from_reader<R>(r: &mut R) -> Result<Self, Self::Error>
    where
        R: bitstream_io::ByteRead + ?Sized,
    {
        let file_size = match r.parse::<Header>()? {
            Header {
                // RIFF identifier
                id: [0x52, 0x49, 0x46, 0x46],
                size,
            } => Ok(size),
            _ => Err(Error::InvalidWave),
        }?;

        // WAVE identifier
        if r.read::<[u8; 4]>()? == [0x57, 0x41, 0x56, 0x45] {
            Ok(Self { file_size })
        } else {
            Err(Error::InvalidWave)
        }
    }
}

impl bitstream_io::ToByteStream for FileHeader {
    type Error = std::io::Error;

    fn to_writer<W>(&self, w: &mut W) -> Result<(), Self::Error>
    where
        W: bitstream_io::ByteWrite + ?Sized,
    {
        w.build(&Header {
            id: [0x52, 0x49, 0x46, 0x46],
            size: self.file_size,
        })?;
        w.write([0x57, 0x41, 0x56, 0x45])?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
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
}

impl bitstream_io::FromByteStream for Fmt {
    type Error = Error;

    fn from_reader<R>(r: &mut R) -> Result<Self, Self::Error>
    where
        R: bitstream_io::ByteRead + ?Sized,
    {
        match r.read::<u16>()? {
            0x0001 => Ok(Self::Standard {
                channels: r.read()?,
                sample_rate: r.read()?,
                data_rate: r.read()?,
                data_block_size: r.read()?,
                bits_per_sample: r.read()?,
            }),
            0xFFFE => {
                let channels = r.read()?;
                let sample_rate = r.read()?;
                let data_rate = r.read()?;
                let data_block_size = r.read()?;
                let bits_per_sample = r.read()?;

                // extension size should be 22 bytes
                if r.read::<u16>()? != 22 {
                    return Err(Error::InvalidWave);
                }

                Ok(Self::Extensible {
                    channels,
                    sample_rate,
                    data_rate,
                    data_block_size,
                    bits_per_sample,
                    valid_bits: r.read()?,
                    channel_mask: r.read()?,
                    sub_format: r.read()?,
                })
            }
            _ => Err(Error::UnsupportedFormat),
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

enum Chunk {
    Header(FileHeader),
    Fmt(Fmt),
    Data { header: Header },
    Foreign { header: Header, data: Vec<u8> },
}

impl Chunk {
    fn is_foreign(&self) -> bool {
        matches!(self, Self::Foreign { .. })
    }
}

impl From<Chunk> for Application {
    fn from(chunk: Chunk) -> Self {
        use bitstream_io::{ByteWrite, ByteWriter, LittleEndian};

        match chunk {
            Chunk::Header(header) => {
                let mut application_data = ByteWriter::endian(vec![], LittleEndian);
                application_data.build(&header).unwrap();

                Application {
                    id: Application::RIFF,
                    data: application_data.into_writer(),
                }
            }
            Chunk::Fmt(fmt) => {
                let mut application_data = ByteWriter::endian(vec![], LittleEndian);
                application_data.build(&fmt).unwrap();

                Application {
                    id: Application::RIFF,
                    data: application_data.into_writer(),
                }
            }
            Chunk::Data { header } => {
                let mut application_data = ByteWriter::endian(vec![], LittleEndian);
                application_data.build(&header).unwrap();

                Application {
                    id: Application::RIFF,
                    data: application_data.into_writer(),
                }
            }
            Chunk::Foreign { header, data } => {
                let mut application_data = ByteWriter::endian(vec![], LittleEndian);
                application_data.build(&header).unwrap();
                application_data.write_bytes(&data).unwrap();
                if data.len() % 2 == 1 {
                    application_data.write::<u8>(0).unwrap();
                }

                Application {
                    id: Application::RIFF,
                    data: application_data.into_writer(),
                }
            }
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
