// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A library for reading, writing, and editing the metadata
//! of FLAC-formatted audio files.
//!
//! ## FLAC Format Layout
//!
//! In order to understand the layout of this crate, it may be
//! helpful to understand the layout of the FLAC format itself.
//! A FLAC file begins with a 4 byte `fLaC` tag
//! (bytes `[0x66, 0x4c, 0x61, 0x43]`) and is followed
//! by one or more metadata blocks.
//! Metadata blocks contain information about the file
//! like stream parameters, textual information, the
//! seek table, etc.
//! The metadata blocks are then followed by one or more
//! frames containing audio data.
//! A typical FLAC file contains only a handful of
//! metadata blocks, but hundreds of frames.
//!
//! Metadata blocks are detailed in the [`metadata`] module,
//! while frames are detailed in [`stream`].
//!
//! ```text
//! ┌──────────┬────────┬┄┄┄┄┄┄┄┄┬┄┄┄┬────────┬┄┄┄┄┄┄┄┄┬┄┄┄╮
//! │ FLAC Tag │ Block₀ │ Block₁ ┆ … ┆ Frame₀ │ Frame₁ ┆ … ┆
//! └──────────┴────────┴┄┄┄┄┄┄┄┄┴┄┄┄┴────────┴┄┄┄┄┄┄┄┄┴┄┄┄╯
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod audio;
pub mod byteorder;
mod crc;
pub mod decode;
pub mod encode;
pub mod metadata;
pub mod stream;

/// A unified FLAC error
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A general I/O error from the underlying stream
    Io(std::io::Error),
    /// A UTF-8 formatting error
    Utf8(Box<std::string::FromUtf8Error>),
    /// A FLAC file missing its initial "fLaC" file tag
    MissingFlacTag,
    /// A FLAC file missing its initial STREAMINFO block
    MissingStreaminfo,
    /// A FLAC file containing multiple STREAMINFO blocks
    MultipleStreaminfo,
    /// A FLAC file containing multiple SEEKTABLE blocks
    MultipleSeekTable,
    /// A FLAC file containing multiple VORBIS_COMMENT blocks
    MultipleVorbisComment,
    /// A SEEKTABLE block whose size isn't evenly divisible
    /// by a whole of number of seek points.
    InvalidSeekTableSize,
    /// A SEEKTABLE point whose offset does not increment properly
    InvalidSeekTablePoint,
    /// A CUESHEET-specific error
    Cuesheet(metadata::CuesheetError),
    /// An undefined PICTURE type
    InvalidPictureType,
    /// Multiple 32x32 PNG icons defined
    MultiplePngIcon,
    /// Multiple general file icons defined
    MultipleGeneralIcon,
    /// A reserved metadata block encountered
    ReservedMetadataBlock,
    /// An invalid metadata block encountered
    InvalidMetadataBlock,
    /// A metadata block's contents are smaller than the size
    /// indicated in the metadata block header.
    InvalidMetadataBlockSize,
    /// An APPLICATION metadata block which is not large enough
    /// to hold any contents beyond its ID.
    InsufficientApplicationBlock,
    /// A `VorbisComment` struct with more entries that can fit in a `u32`
    ExcessiveVorbisEntries,
    /// A `VorbisComment` or `Picture` struct with strings longer than a `u32`
    ExcessiveStringLength,
    /// A `Picture` struct whose data is larger than a `u32`
    ExcessivePictureSize,
    /// A block size less than 15 that's not the last block
    ShortBlock,
    /// A metadata block larger than its 24-bit size field can hold
    ExcessiveBlockSize,
    /// Invalid frame sync code
    InvalidSyncCode,
    /// Invalid frame block size
    InvalidBlockSize,
    /// Block size in frame is larger than maximum block size in STREAMINFO
    BlockSizeMismatch,
    /// Invalid frame sample rate
    InvalidSampleRate,
    /// Non-subset frame sample rate
    NonSubsetSampleRate,
    /// Non-subset frame bits-per-sample
    NonSubsetBitsPerSample,
    /// Mismatch between frame sample rate and STREAMINFO sample rate
    SampleRateMismatch,
    /// Excessive channel count
    ExcessiveChannels,
    /// Invalid frame channel assignment
    InvalidChannels,
    /// Channel count in frame differs from channel count in STREAMINFO
    ChannelsMismatch,
    /// Invalid frame bits-per-sample
    InvalidBitsPerSample,
    /// Excessive number of bits-per-sample
    ExcessiveBps,
    /// Bits-per-sample in frame differs from bits-per-sample in STREAMINFO
    BitsPerSampleMismatch,
    /// Invalid frame number
    InvalidFrameNumber,
    /// Excessive frame number
    ExcessiveFrameNumber,
    /// CRC-8 mismatch in frame header
    Crc8Mismatch,
    /// CRC-16 mismatch in frame footer
    Crc16Mismatch,
    /// Invalid subframe header
    InvalidSubframeHeader,
    /// Invalid subframe header type
    InvalidSubframeHeaderType,
    /// Excessive number of wasted bits-per-sample
    ExcessiveWastedBits,
    /// Insufficient number of residuals in residuals block
    MissingResiduals,
    /// Invalid residual coding method
    InvalidCodingMethod,
    /// Invalid residual partition order
    InvalidPartitionOrder,
    /// Invalid FIXED subframe predictor order
    InvalidFixedOrder,
    /// Invalid LPC subframe predictor order
    InvalidLpcOrder,
    /// Invalid coefficient precision bits
    InvalidQlpPrecision,
    /// Negative shift value in LPC subframe
    NegativeLpcShift,
    /// Accumulator overflow in LPC subframe
    AccumulatorOverflow,
    /// Too many samples encountered in stream
    TooManySamples,
    /// Too many samples requested by encoder
    ExcessiveTotalSamples,
    /// No samples written by encoder
    NoSamples,
    /// Number of samples written to stream differs from expected
    SampleCountMismatch,
    /// Residual overflow
    ResidualOverflow,
    /// Number of samples not evenly divisible by number of channels
    SamplesNotDivisibleByChannels,
    /// Invalid total byte count
    InvalidTotalBytes,
    /// Invalid total samples count
    InvalidTotalSamples,
}

impl From<std::io::Error> for Error {
    #[inline]
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<std::string::FromUtf8Error> for Error {
    #[inline]
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::Utf8(Box::new(error))
    }
}

impl From<metadata::CuesheetError> for Error {
    fn from(error: metadata::CuesheetError) -> Self {
        Self::Cuesheet(error)
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Utf8(e) => e.fmt(f),
            Self::Cuesheet(e) => e.fmt(f),
            Self::MissingFlacTag => "missing FLAC tag".fmt(f),
            Self::MissingStreaminfo => "STREAMINFO block not first in file".fmt(f),
            Self::MultipleStreaminfo => "multiple STREAMINFO blocks found in file".fmt(f),
            Self::MultipleSeekTable => "multiple SEEKTABLE blocks found in file".fmt(f),
            Self::MultipleVorbisComment => "multiple VORBIS_COMMENT blocks found in file".fmt(f),
            Self::InvalidSeekTableSize => "invalid SEEKTABLE block size".fmt(f),
            Self::InvalidSeekTablePoint => "invalid SEEKTABLE point".fmt(f),
            Self::InvalidPictureType => "reserved PICTURE type".fmt(f),
            Self::MultiplePngIcon => "multiple PNG icons in PICTURE blocks".fmt(f),
            Self::MultipleGeneralIcon => "multiple general file icons in PICTURE blocks".fmt(f),
            Self::ReservedMetadataBlock => "reserved metadata block".fmt(f),
            Self::InvalidMetadataBlock => "invalid metadata block".fmt(f),
            Self::InvalidMetadataBlockSize => "invalid metadata block size".fmt(f),
            Self::InsufficientApplicationBlock => "APPLICATION block too small for data".fmt(f),
            Self::ExcessiveVorbisEntries => "excessive number of VORBIS_COMMENT entries".fmt(f),
            Self::ExcessiveStringLength => "excessive string length".fmt(f),
            Self::ExcessivePictureSize => "excessive PICTURE data size".fmt(f),
            Self::ExcessiveBlockSize => "excessive metadata block size".fmt(f),
            Self::InvalidSyncCode => "invalid frame sync code".fmt(f),
            Self::InvalidBlockSize => "invalid frame block size".fmt(f),
            Self::ShortBlock => "block size <= 14 must be last in stream".fmt(f),
            Self::BlockSizeMismatch => {
                "block size in frame larger than maximum block size in STREAMINFO".fmt(f)
            }
            Self::InvalidSampleRate => "invalid frame sample rate".fmt(f),
            Self::NonSubsetSampleRate => "sample rate undefined for subset stream".fmt(f),
            Self::NonSubsetBitsPerSample => "bits-per-sample undefined for subset stream".fmt(f),
            Self::SampleRateMismatch => {
                "sample rate in frame differs from sample rate in STREAMINFO".fmt(f)
            }
            Self::ExcessiveChannels => "excessive channel count".fmt(f),
            Self::InvalidChannels => "invalid frame channel assignment".fmt(f),
            Self::ChannelsMismatch => {
                "channel count in frame differs from channel count in STREAMINFO".fmt(f)
            }
            Self::InvalidBitsPerSample => "invalid frame bits-per-sample".fmt(f),
            Self::ExcessiveBps => "bits-per-sample higher than 32".fmt(f),
            Self::BitsPerSampleMismatch => {
                "bits-per-sample in frame differs from bits-per-sample in STREAMINFO".fmt(f)
            }
            Self::InvalidFrameNumber => "invalid frame number".fmt(f),
            Self::ExcessiveFrameNumber => "excessive frame number".fmt(f),
            Self::Crc8Mismatch => "CRC-8 mismatch in frame header".fmt(f),
            Self::Crc16Mismatch => "CRC-16 mismatch in frame footer".fmt(f),
            Self::InvalidSubframeHeader => "invalid subframe header".fmt(f),
            Self::InvalidSubframeHeaderType => "invalid subframe header type".fmt(f),
            Self::ExcessiveWastedBits => "excessive number of wasted BPS".fmt(f),
            Self::MissingResiduals => "insufficient number of residuals".fmt(f),
            Self::InvalidCodingMethod => "invalid residual coding method".fmt(f),
            Self::InvalidPartitionOrder => "invalid residual partition order".fmt(f),
            Self::InvalidFixedOrder => "invalid FIXED subframe predictor order".fmt(f),
            Self::InvalidLpcOrder => "invalid LPC subframe predictor order".fmt(f),
            Self::InvalidQlpPrecision => "invalid QLP precision bits".fmt(f),
            Self::NegativeLpcShift => "negative shift in LPC subframe".fmt(f),
            Self::AccumulatorOverflow => "accumulator overflow in LPC subframe".fmt(f),
            Self::TooManySamples => "more samples in stream than indicated in STREAMINFO".fmt(f),
            Self::ExcessiveTotalSamples => "too many samples requested".fmt(f),
            Self::NoSamples => "no samples written to encoder".fmt(f),
            Self::SampleCountMismatch => "samples written to stream differ from expected".fmt(f),
            Self::ResidualOverflow => "residual value too large".fmt(f),
            Self::SamplesNotDivisibleByChannels => {
                "number of samples not divisible number number of channels".fmt(f)
            }
            Self::InvalidTotalBytes => "invalid total byte count".fmt(f),
            Self::InvalidTotalSamples => "invalid total samples count".fmt(f),
        }
    }
}

impl From<Error> for std::io::Error {
    fn from(err: Error) -> Self {
        match err {
            Error::Io(io) => io,
            Error::Utf8(e) => std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            other => std::io::Error::new(std::io::ErrorKind::InvalidData, other.to_string()),
        }
    }
}

struct Counter<F> {
    stream: F,
    count: u64,
}

impl<F> Counter<F> {
    fn new(stream: F) -> Self {
        Self { stream, count: 0 }
    }

    fn stream(&mut self) -> &mut F {
        &mut self.stream
    }
}

impl<F: std::io::Read> std::io::Read for Counter<F> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf).inspect(|bytes| {
            self.count += u64::try_from(*bytes).unwrap();
        })
    }
}

impl<F: std::io::Write> std::io::Write for Counter<F> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf).inspect(|bytes| {
            self.count += u64::try_from(*bytes).unwrap();
        })
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}
