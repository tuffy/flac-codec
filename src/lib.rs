// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A library for reading, writing, and editing the metadata
//! of FLAC-formatted audio files.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod metadata;
pub mod stream;

/// A unified FLAC format error
#[derive(Debug)]
pub enum Error {
    /// A general I/O error from the underlying stream
    Io(std::io::Error),
    /// A UTF-8 formatting error
    Utf8(std::string::FromUtf8Error),
    /// A FLAC file missing its initial "fLaC" file tag
    MissingFlacTag,
    /// A FLAC file missing its initial STREAMINFO block
    MissingStreaminfo,
    /// A FLAC file containing multiple STREAMINFO blocks
    MultipleStreaminfo,
    /// A FLAC file containing multiple SEEKTABLE blocks
    MultipleSeekTable,
    /// A SEEKTABLE block whose size isn't evenly divisible
    /// by a whole of number of seek points.
    InvalidSeekTableSize,
    /// A SEEKTABLE point whose offset does not increment properly
    InvalidSeekTablePoint,
    /// A CDDA CUESHEET offset that does not start on a CD frame boundary
    InvalidCuesheetOffset,
    /// An invalid CDDA CUESHEET track number
    InvalidCuesheetTrackNumber,
    /// A non-lead-out CUESHEET track containing no index points,
    /// or a lead-out CUESHEET track containing some index points.
    InvalidCuesheetIndexPoints,
    /// A CUESHEET track index point that does not start with 0 or 1,
    /// or does not increment by 1.
    InvalidCuesheetIndexPointNum,
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
    /// A `Cuesheet` struct with more than `u8` tracks
    ExcessiveCuesheetTracks,
    /// A `CuesheetTrack` struct with more than `u8` index points
    ExcessiveCuesheetIndexPoints,
    /// A metadata block larger than its 24-bit size field can hold
    ExcessiveBlockSize,
    /// Invalid frame sync code
    InvalidSyncCode,
    /// Invalid frame block size
    InvalidBlockSize,
    /// Invalid frame sample rate
    InvalidSampleRate,
    /// Invalid frame channel assignment
    InvalidChannels,
    /// Invalid frame bits-per-sample
    InvalidBitsPerSample,
    /// Invalid frame number
    InvalidFrameNumber,
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::Utf8(error)
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Utf8(e) => e.fmt(f),
            Self::MissingFlacTag => "missing FLAC tag".fmt(f),
            Self::MissingStreaminfo => "STREAMINFO block not first in file".fmt(f),
            Self::MultipleStreaminfo => "multiple STREAMINFO blocks found in file".fmt(f),
            Self::MultipleSeekTable => "multiple SEEKTABLE blocks found in file".fmt(f),
            Self::InvalidSeekTableSize => "invalid SEEKTABLE block size".fmt(f),
            Self::InvalidSeekTablePoint => "invalid SEEKTABLE point".fmt(f),
            Self::InvalidCuesheetOffset => "invalid CUESHEET sample offset".fmt(f),
            Self::InvalidCuesheetTrackNumber => "invalid CUESHEET track number".fmt(f),
            Self::InvalidCuesheetIndexPoints => {
                "invalid number of CUESHEET track index points".fmt(f)
            }
            Self::InvalidCuesheetIndexPointNum => "invalid CUESHEET index point number".fmt(f),
            Self::InvalidPictureType => "reserved PICTURE type".fmt(f),
            Self::MultiplePngIcon => "multiple PNG icons in PICTURE blocks".fmt(f),
            Self::MultipleGeneralIcon => "multiple general file icons in PICTURE blocks".fmt(f),
            Self::ReservedMetadataBlock => "reserved metadata block".fmt(f),
            Self::InvalidMetadataBlock => "invalid metadata block".fmt(f),
            Self::InvalidMetadataBlockSize => "invalid metadat block size".fmt(f),
            Self::InsufficientApplicationBlock => "APPLICATION block too small for data".fmt(f),
            Self::ExcessiveVorbisEntries => "excessive number of VORBIS_COMMENT entries".fmt(f),
            Self::ExcessiveStringLength => "excessive string length".fmt(f),
            Self::ExcessivePictureSize => "excessive PICTURE data size".fmt(f),
            Self::ExcessiveCuesheetTracks => "excessuve number of CUESHEET tracks".fmt(f),
            Self::ExcessiveCuesheetIndexPoints => {
                "excessuve number of CUESHEET track index points".fmt(f)
            }
            Self::ExcessiveBlockSize => "excessive metadata block size".fmt(f),
            Self::InvalidSyncCode => "invalid frame sync code".fmt(f),
            Self::InvalidBlockSize => "invalid frame block size".fmt(f),
            Self::InvalidSampleRate => "invalid frame sample rate".fmt(f),
            Self::InvalidChannels => "invalid frame channel assignment".fmt(f),
            Self::InvalidBitsPerSample => "invalid frame bits-per-sample".fmt(f),
            Self::InvalidFrameNumber => "invalid frame numbe".fmt(f),
        }
    }
}
