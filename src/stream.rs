// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For handling common FLAC stream items

use crate::Error;
use crate::metadata::Streaminfo;
use bitstream_io::{BitRead, FromBitStream, FromBitStreamWith};
use std::num::NonZero;

/// A FLAC frame header
#[derive(Debug)]
pub struct FrameHeader {
    /// The blocking strategy bit
    pub blocking_strategy: bool,
    /// The block size, in samples
    pub block_size: u16,
    /// The sample rate, in Hz
    pub sample_rate: u32,
    /// How the channels are assigned
    pub channel_assignment: ChannelAssignment,
    /// The number if bits per output sample
    pub bits_per_sample: u8,
    /// The frame's number in the stream
    pub frame_number: u32,
}

impl FrameHeader {
    /// Reads new header from the given reader
    pub fn read<R: std::io::Read>(reader: &mut R, streaminfo: &Streaminfo) -> Result<Self, Error> {
        use crate::crc::{Checksum, Crc8, CrcReader};
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc8: CrcReader<_, Crc8> = CrcReader::new(reader);
        BitReader::endian(crc8.by_ref(), BigEndian)
            .parse_with(streaminfo)
            .and_then(|header| {
                crc8.into_checksum()
                    .valid()
                    .then_some(header)
                    .ok_or(Error::Crc8Mismatch)
            })
    }
}

impl FromBitStreamWith<'_> for FrameHeader {
    type Error = Error;
    type Context = Streaminfo;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        streaminfo: &Streaminfo,
    ) -> Result<Self, Self::Error> {
        if r.read::<15, u16>()? != 0b111111111111100 {
            return Err(Error::InvalidSyncCode);
        }
        let blocking_strategy = r.read_bit()?;
        let encoded_block_size = r.read::<4, u8>()?;
        let encoded_sample_rate = r.read::<4, u8>()?;
        let encoded_channels = r.read::<4, u8>()?;
        let encoded_bps = r.read::<3, u8>()?;
        r.skip(1)?;
        let frame_number = read_frame_number(r)?;

        let block_size = match encoded_block_size {
            0b0000 => return Err(Error::InvalidBlockSize),
            0b0001 => 192,
            v @ 0b0010..=0b0101 => 144 * (1 << v),
            0b0110 => r.read::<8, u16>()? + 1,
            0b0111 => r.read::<16, u16>()? + 1,
            v @ 0b1000..=0b1111 => 1 << v,
            _ => unreachable!(), // 4-bit field
        };

        if block_size > streaminfo.maximum_block_size {
            return Err(Error::BlockSizeMismatch);
        }

        let sample_rate = match encoded_sample_rate {
            0b0000 => streaminfo.sample_rate,
            0b0001 => 88200,
            0b0010 => 176400,
            0b0011 => 192000,
            0b0100 => 8000,
            0b0101 => 16000,
            0b0110 => 22050,
            0b0111 => 24000,
            0b1000 => 32000,
            0b1001 => 44100,
            0b1010 => 48000,
            0b1011 => 96000,
            0b1100 => r.read::<8, u32>()? * 1000,
            0b1101 => r.read::<16, _>()?,
            0b1110 => r.read::<16, u32>()? * 10,
            0b1111 => return Err(Error::InvalidSampleRate),
            _ => unreachable!(), // 4-bit field
        };

        if sample_rate != streaminfo.sample_rate {
            return Err(Error::SampleRateMismatch);
        }

        let channel_assignment = match encoded_channels {
            c @ 0b0000..=0b0111 => ChannelAssignment::Independent(c + 1),
            0b1000 => ChannelAssignment::LeftSide,
            0b1001 => ChannelAssignment::SideRight,
            0b1010 => ChannelAssignment::MidSide,
            0b1011..=0b1111 => return Err(Error::InvalidChannels),
            _ => unreachable!(), // 4-bit field
        };

        if channel_assignment.count() != streaminfo.channels.get() {
            return Err(Error::ChannelsMismatch);
        }

        let bits_per_sample = match encoded_bps {
            0b000 => streaminfo.bits_per_sample.get(),
            0b001 => 8,
            0b010 => 12,
            0b011 => return Err(Error::InvalidBitsPerSample),
            0b100 => 16,
            0b101 => 20,
            0b110 => 24,
            0b111 => 32,
            _ => unreachable!(), // 3-bit field
        };

        if bits_per_sample != streaminfo.bits_per_sample.get() {
            return Err(Error::BitsPerSampleMismatch);
        }

        r.skip(8)?; // CRC-8

        Ok(Self {
            blocking_strategy,
            frame_number,
            block_size,
            sample_rate,
            channel_assignment,
            bits_per_sample,
        })
    }
}

/// How the channels are assigned in a FLAC frame
#[derive(Debug)]
pub enum ChannelAssignment {
    /// Channels are stored independently
    Independent(u8),
    /// Channel 0 is stored verbatim, channel 1 derived from both
    LeftSide,
    /// Channel 0 is derived from both, channel 1 is stored verbatim
    SideRight,
    /// Channel 0 is averaged from both, channel 1 is derived from both
    MidSide,
}

impl ChannelAssignment {
    /// Returns total number of channels defined by assignment
    pub fn count(&self) -> u8 {
        match self {
            Self::Independent(c) => *c,
            _ => 2,
        }
    }
}

fn read_frame_number<R: BitRead + ?Sized>(r: &mut R) -> Result<u32, Error> {
    match r.read_unary::<0>()? {
        0 => Ok(r.read::<7, _>()?),
        1 => Err(Error::InvalidFrameNumber),
        bytes @ 2..=7 => {
            let mut frame = r.read_var(7 - bytes)?;
            for _ in 1..bytes {
                match r.read::<2, u8>()? {
                    0b10 => {
                        frame = (frame << 6) | r.read::<6, u32>()?;
                    }
                    _ => return Err(Error::InvalidFrameNumber),
                }
            }
            Ok(frame)
        }
        _ => Err(Error::InvalidFrameNumber),
    }
}

/// A Subframe Header
#[derive(Debug)]
pub struct SubframeHeader {
    /// The subframe header's type
    pub type_: SubframeHeaderType,
    /// The number of wasted bits-per-sample,
    pub wasted_bps: u32,
}

impl FromBitStream for SubframeHeader {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read_bit()? {
            false => Ok(Self {
                type_: r.parse()?,
                wasted_bps: match r.read_bit()? {
                    false => 0,
                    true => r.read_unary::<1>()? + 1,
                },
            }),
            true => Err(Error::InvalidSubframeHeader),
        }
    }
}

/// A subframe header's type
#[derive(Debug)]
pub enum SubframeHeaderType {
    /// All samples are the same
    Constant,
    /// All samples as stored verbatim, without compression
    Verbatim,
    /// Samples are stored with one of a set of fixed LPC parameters
    Fixed(u8),
    /// Samples are stored with dynamic LPC parameters
    Lpc(NonZero<u8>),
}

impl FromBitStream for SubframeHeaderType {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read::<6, u8>()? {
            0b000000 => Ok(Self::Constant),
            0b000001 => Ok(Self::Verbatim),
            v @ 0b001000..=0b001100 => Ok(Self::Fixed(v - 8)),
            v @ 0b100000..=0b111111 => Ok(Self::Lpc(NonZero::new(v - 31).unwrap())),
            _ => Err(Error::InvalidSubframeHeaderType),
        }
    }
}
