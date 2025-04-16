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
use bitstream_io::{BitCount, BitRead, BitWrite, FromBitStream, FromBitStreamWith, ToBitStream};
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
    pub bits_per_sample: BitCount<32>,
    /// The frame's number in the stream
    pub frame_number: FrameNumber,
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

    fn parse<R: BitRead + ?Sized>(
        r: &mut R,
        non_subset_rate: impl FnOnce() -> Result<u32, Error>,
        non_subset_bps: impl FnOnce() -> Result<BitCount<32>, Error>,
    ) -> Result<Self, Error> {
        r.read_const::<15, 0b111111111111100, _>(Error::InvalidSyncCode)?;
        let blocking_strategy = r.read_bit()?;
        let encoded_block_size = r.read::<4, u8>()?;
        let encoded_sample_rate = r.read::<4, u8>()?;
        let encoded_channels = r.read::<4, u8>()?;
        let encoded_bps = r.read::<3, u8>()?;
        r.skip(1)?;
        let frame_number = r.parse()?;

        let frame_header = Self {
            blocking_strategy,
            frame_number,
            block_size: match encoded_block_size {
                0b0000 => return Err(Error::InvalidBlockSize),
                0b0001 => 192,
                v @ 0b0010..=0b0101 => 144 * (1 << v),
                0b0110 => r.read::<8, u16>()? + 1,
                0b0111 => r.read::<16, u16>()? + 1,
                v @ 0b1000..=0b1111 => 1 << v,
                _ => unreachable!(), // 4-bit field
            },
            sample_rate: match encoded_sample_rate {
                0b0000 => non_subset_rate()?,
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
            },
            channel_assignment: match encoded_channels {
                c @ 0b0000..=0b0111 => ChannelAssignment::Independent(c + 1),
                0b1000 => ChannelAssignment::LeftSide,
                0b1001 => ChannelAssignment::SideRight,
                0b1010 => ChannelAssignment::MidSide,
                0b1011..=0b1111 => return Err(Error::InvalidChannels),
                _ => unreachable!(), // 4-bit field
            },
            bits_per_sample: match encoded_bps {
                0b000 => non_subset_bps()?,
                0b001 => BitCount::new::<8>(),
                0b010 => BitCount::new::<12>(),
                0b011 => return Err(Error::InvalidBitsPerSample),
                0b100 => BitCount::new::<16>(),
                0b101 => BitCount::new::<20>(),
                0b110 => BitCount::new::<24>(),
                0b111 => BitCount::new::<32>(),
                _ => unreachable!(), // 3-bit field
            },
        };

        r.skip(8)?; // CRC-8

        Ok(frame_header)
    }
}

impl FromBitStreamWith<'_> for FrameHeader {
    type Error = Error;
    type Context = Streaminfo;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        streaminfo: &Streaminfo,
    ) -> Result<Self, Self::Error> {
        FrameHeader::parse(
            r,
            || Ok(streaminfo.sample_rate),
            || Ok(streaminfo.bits_per_sample),
        )
        .and_then(|h| {
            (h.block_size <= streaminfo.maximum_block_size)
                .then_some(h)
                .ok_or(Error::BlockSizeMismatch)
        })
        .and_then(|h| {
            (h.sample_rate == streaminfo.sample_rate)
                .then_some(h)
                .ok_or(Error::SampleRateMismatch)
        })
        .and_then(|h| {
            (h.channel_assignment.count() == streaminfo.channels.get())
                .then_some(h)
                .ok_or(Error::ChannelsMismatch)
        })
        .and_then(|h| {
            (h.bits_per_sample == streaminfo.bits_per_sample)
                .then_some(h)
                .ok_or(Error::BitsPerSampleMismatch)
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

/// A frame number in the stream, as FLAC frames
#[derive(Debug)]
pub struct FrameNumber(pub u64);

impl FromBitStream for FrameNumber {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read_unary::<0>()? {
            0 => Ok(Self(r.read::<7, _>()?)),
            1 => Err(Error::InvalidFrameNumber),
            bytes @ 2..=7 => {
                let mut frame = r.read_var(7 - bytes)?;
                for _ in 1..bytes {
                    r.read_const::<2, 0b10, _>(Error::InvalidFrameNumber)?;
                    frame = (frame << 6) | r.read::<6, u64>()?;
                }
                Ok(Self(frame))
            }
            _ => Err(Error::InvalidFrameNumber),
        }
    }
}

impl ToBitStream for FrameNumber {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        #[inline]
        fn byte(num: u64, byte: u32) -> u8 {
            0b10_000000 | ((num >> (6 * byte)) & 0b111111) as u8
        }

        match self.0 {
            v @ 0..=0x7F => {
                w.write_unary::<0>(0)?;
                w.write::<7, _>(v)?;
                Ok(())
            }
            v @ 0x80..=0x7FF => {
                w.write_unary::<0>(2)?;
                w.write::<5, _>(v >> 6)?;
                w.write::<8, _>(byte(v, 0))?;
                Ok(())
            }
            v @ 0x800..=0xFFFF => {
                w.write_unary::<0>(3)?;
                w.write::<4, _>(v >> (6 * 2))?;
                w.write::<8, _>(byte(v, 1))?;
                w.write::<8, _>(byte(v, 0))?;
                Ok(())
            }
            v @ 0x1_0000..=0x1F_FFFF => {
                w.write_unary::<0>(4)?;
                w.write::<3, _>(v >> (6 * 3))?;
                w.write::<8, _>(byte(v, 2))?;
                w.write::<8, _>(byte(v, 1))?;
                w.write::<8, _>(byte(v, 0))?;
                Ok(())
            }
            v @ 0x20_0000..=0x3FF_FFFF => {
                w.write_unary::<0>(5)?;
                w.write::<2, _>(v >> (6 * 4))?;
                w.write::<8, _>(byte(v, 3))?;
                w.write::<8, _>(byte(v, 2))?;
                w.write::<8, _>(byte(v, 1))?;
                w.write::<8, _>(byte(v, 0))?;
                Ok(())
            }
            v @ 0x400_0000..=0x7FFF_FFFF => {
                w.write_unary::<0>(6)?;
                w.write::<1, _>(v >> (6 * 5))?;
                w.write::<8, _>(byte(v, 4))?;
                w.write::<8, _>(byte(v, 3))?;
                w.write::<8, _>(byte(v, 2))?;
                w.write::<8, _>(byte(v, 1))?;
                w.write::<8, _>(byte(v, 0))?;
                Ok(())
            }
            v @ 0x8000_0000..=0xF_FFFF_FFFF => {
                w.write_unary::<0>(7)?;
                w.write::<8, _>(byte(v, 5))?;
                w.write::<8, _>(byte(v, 4))?;
                w.write::<8, _>(byte(v, 3))?;
                w.write::<8, _>(byte(v, 2))?;
                w.write::<8, _>(byte(v, 1))?;
                w.write::<8, _>(byte(v, 0))?;
                Ok(())
            }
            _ => Err(Error::InvalidFrameNumber),
        }
    }
}

#[test]
fn test_frame_number() {
    use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};

    let mut buf: [u8; 7] = [0; 7];

    for i in (0..=0xFFFF)
        .chain((0x1_0000..=0x1F_FFFF).step_by(32))
        .chain((0x20_0000..=0x3FF_FFFF).step_by(1024))
        .chain((0x400_0000..=0x7FFF_FFFF).step_by(33760))
        .chain((0x8000_0000..=0xF_FFFF_FFFF).step_by(1048592))
    {
        let num = FrameNumber(i);

        assert!(
            BitWriter::endian(buf.as_mut_slice(), BigEndian)
                .build(&num)
                .is_ok()
        );

        let num2 = BitReader::endian(buf.as_slice(), BigEndian)
            .parse::<FrameNumber>()
            .unwrap();

        assert_eq!(num.0, num2.0);

        buf.fill(0);
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
        r.read_const::<1, 0, _>(Error::InvalidSubframeHeader)?;
        Ok(Self {
            type_: r.parse()?,
            wasted_bps: match r.read_bit()? {
                false => 0,
                true => r.read_unary::<1>()? + 1,
            },
        })
    }
}

impl ToBitStream for SubframeHeader {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        w.write_const::<1, 0>()?;
        w.build(&self.type_)?;
        match self.wasted_bps.checked_sub(1) {
            None => w.write_bit(false)?,
            Some(wasted) => {
                w.write_bit(true)?;
                w.write_unary::<1>(wasted - 1)?;
            }
        }

        Ok(())
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
    Fixed(&'static [i64]),
    /// Samples are stored with dynamic LPC parameters
    Lpc(NonZero<u8>),
}

impl FromBitStream for SubframeHeaderType {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read::<6, u8>()? {
            0b000000 => Ok(Self::Constant),
            0b000001 => Ok(Self::Verbatim),
            0b001000 => Ok(Self::Fixed(&[])),
            0b001001 => Ok(Self::Fixed(&[1])),
            0b001010 => Ok(Self::Fixed(&[2, -1])),
            0b001011 => Ok(Self::Fixed(&[3, -3, 1])),
            0b001100 => Ok(Self::Fixed(&[4, -6, 4, -1])),
            v @ 0b100000..=0b111111 => Ok(Self::Lpc(NonZero::new(v - 31).unwrap())),
            _ => Err(Error::InvalidSubframeHeaderType),
        }
    }
}

impl ToBitStream for SubframeHeaderType {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        w.write::<6, u8>(match self {
            Self::Constant => 0b000000,
            Self::Verbatim => 0b000001,
            Self::Fixed(coeffs) => 0b001000 + coeffs.len() as u8,
            Self::Lpc(order) => order.get() + 31,
        })?;
        Ok(())
    }
}
