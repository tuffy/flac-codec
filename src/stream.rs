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
use bitstream_io::{
    BitCount, BitRead, BitWrite, FromBitStream, FromBitStreamWith, SignedBitCount, ToBitStream,
    ToBitStreamWith,
};
use std::num::NonZero;

/// A FLAC frame header
#[derive(Debug)]
pub struct FrameHeader {
    /// The blocking strategy bit
    pub blocking_strategy: bool,
    /// The block size, in samples
    pub block_size: BlockSize<u16>,
    /// The sample rate, in Hz
    pub sample_rate: SampleRate<u32>,
    /// How the channels are assigned
    pub channel_assignment: ChannelAssignment,
    /// The number if bits per output sample
    pub bits_per_sample: SignedBitCount<32>,
    /// The frame's number in the stream
    pub frame_number: FrameNumber,
}

impl FrameHeader {
    const SYNC_CODE: u32 = 0b111111111111100;

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

    /// Builds header to the given writer
    pub fn write<W: std::io::Write>(
        &self,
        writer: &mut W,
        streaminfo: &Streaminfo,
    ) -> Result<(), Error> {
        use crate::crc::{Crc8, CrcWriter};
        use bitstream_io::{BigEndian, BitWriter};
        use std::io::Write;

        let mut crc8: CrcWriter<_, Crc8> = CrcWriter::new(writer.by_ref());
        BitWriter::endian(crc8.by_ref(), BigEndian).build_with(self, streaminfo)?;
        let crc8 = crc8.into_checksum().into();
        writer.write_all(std::slice::from_ref(&crc8))?;
        Ok(())
    }

    fn parse<R: BitRead + ?Sized>(
        r: &mut R,
        non_subset_rate: impl FnOnce() -> Result<u32, Error>,
        non_subset_bps: impl FnOnce() -> Result<SignedBitCount<32>, Error>,
    ) -> Result<Self, Error> {
        r.read_const::<15, { Self::SYNC_CODE }, _>(Error::InvalidSyncCode)?;
        let blocking_strategy = r.read_bit()?;
        // let encoded_block_size = r.read::<4, u8>()?;
        let encoded_block_size = r.parse::<BlockSize<()>>()?;
        let encoded_sample_rate = r.parse::<SampleRate<()>>()?;
        let encoded_channels = r.read::<4, u8>()?;
        let encoded_bps = r.read::<3, u8>()?;
        r.skip(1)?;
        let frame_number = r.parse()?;

        let frame_header = Self {
            blocking_strategy,
            frame_number,
            block_size: encoded_block_size.finalize_read(r)?,
            sample_rate: encoded_sample_rate.finalize_read(r, non_subset_rate)?,
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
                0b001 => SignedBitCount::new::<8>(),
                0b010 => SignedBitCount::new::<12>(),
                0b011 => return Err(Error::InvalidBitsPerSample),
                0b100 => SignedBitCount::new::<16>(),
                0b101 => SignedBitCount::new::<20>(),
                0b110 => SignedBitCount::new::<24>(),
                0b111 => SignedBitCount::new::<32>(),
                _ => unreachable!(), // 3-bit field
            },
        };

        r.skip(8)?; // CRC-8

        Ok(frame_header)
    }

    fn build<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        _non_subset_rate: impl FnOnce() -> Result<u32, Error>,
        non_subset_bps: impl FnOnce() -> Result<SignedBitCount<32>, Error>,
    ) -> Result<(), Error> {
        w.write_const::<15, { Self::SYNC_CODE }>()?;
        w.write_bit(self.blocking_strategy)?;
        w.build(&self.block_size)?;
        w.build(&self.sample_rate)?;
        w.write::<4, u8>(match self.channel_assignment {
            ChannelAssignment::Independent(c) => c - 1,
            ChannelAssignment::LeftSide => 0b1000,
            ChannelAssignment::SideRight => 0b1001,
            ChannelAssignment::MidSide => 0b1010,
        })?;
        w.write::<3, u8>(match self.bits_per_sample.into() {
            8 => 0b001,
            12 => 0b010,
            16 => 0b100,
            20 => 0b101,
            24 => 0b110,
            32 => 0b111,
            bps if bps == non_subset_bps()?.into() => 0b000,
            _ => return Err(Error::InvalidBitsPerSample),
        })?;
        w.pad(1)?;
        w.build(&self.frame_number)?;

        // uncommon block size
        match self.block_size {
            BlockSize::Read8(size) => {
                w.write::<8, _>(size.checked_sub(1).ok_or(Error::InvalidBlockSize)?)?
            }
            BlockSize::Read16(size) => {
                w.write::<16, _>(size.checked_sub(1).ok_or(Error::InvalidBlockSize)?)?
            }
            _ => { /* do nothing */ }
        }

        // uncommon sample rate
        match self.sample_rate {
            SampleRate::Read8x1000(rate) => w.write::<8, _>(rate / 1000)?,
            SampleRate::Read16(rate) => {
                w.write::<16, _>(rate)?;
            }
            SampleRate::Read16x10(rate) => {
                w.write::<16, _>(rate / 10)?;
            }
            _ => { /* do nothing */ }
        }

        Ok(())
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
            (u16::from(h.block_size) <= streaminfo.maximum_block_size)
                .then_some(h)
                .ok_or(Error::BlockSizeMismatch)
        })
        .and_then(|h| {
            (u32::from(h.sample_rate) == streaminfo.sample_rate)
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

impl ToBitStreamWith<'_> for FrameHeader {
    type Error = Error;
    type Context = Streaminfo;

    #[inline]
    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        streaminfo: &Streaminfo,
    ) -> Result<(), Self::Error> {
        self.build(
            w,
            || Ok(streaminfo.sample_rate),
            || Ok(streaminfo.bits_per_sample),
        )
    }
}

/// Possible block sizes in a FLAC frame
#[derive(Copy, Clone, Debug)]
pub enum BlockSize<B> {
    /// 192 samples
    Samples192,
    /// 144 * (1 << v) samples
    Samples576to4608(u16),
    /// Read 8 bits + 1
    Read8(B),
    /// Read 16 bits + 1
    Read16(B),
    /// (1 << v) samples
    Samples256to32768(u16),
}

impl FromBitStream for BlockSize<()> {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        match r.read::<4, u8>()? {
            0b0000 => Err(Error::InvalidBlockSize),
            0b0001 => Ok(Self::Samples192),
            v @ 0b0010..=0b0101 => Ok(Self::Samples576to4608(144 * (1 << v))),
            0b0110 => Ok(Self::Read8(())),
            0b0111 => Ok(Self::Read16(())),
            v @ 0b1000..=0b1111 => Ok(Self::Samples256to32768(1 << v)),
            _ => unreachable!(), // 4-bit field
        }
    }
}

impl BlockSize<()> {
    fn finalize_read<R: BitRead + ?Sized>(self, r: &mut R) -> Result<BlockSize<u16>, Error> {
        match self {
            Self::Samples192 => Ok(BlockSize::Samples192),
            Self::Samples576to4608(s) => Ok(BlockSize::Samples576to4608(s)),
            Self::Read8(()) => Ok(BlockSize::Read8(
                r.read::<8, u16>()?
                    .checked_add(1)
                    .ok_or(Error::InvalidBlockSize)?,
            )),
            Self::Read16(()) => Ok(BlockSize::Read16(
                r.read::<16, u16>()?
                    .checked_add(1)
                    .ok_or(Error::InvalidBlockSize)?,
            )),
            Self::Samples256to32768(s) => Ok(BlockSize::Samples256to32768(s)),
        }
    }
}

impl<B> ToBitStream for BlockSize<B> {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<4, u8>(match self {
            Self::Samples192 => 0b0001,
            Self::Samples576to4608(s) => (s / 144).ilog2().try_into().unwrap(),
            Self::Read8(_) => 0b0110,
            Self::Read16(_) => 0b0111,
            Self::Samples256to32768(s) => s.ilog2().try_into().unwrap(),
        })
    }
}

impl From<BlockSize<u16>> for u16 {
    fn from(size: BlockSize<u16>) -> Self {
        match size {
            BlockSize::Samples192 => 192,
            BlockSize::Samples576to4608(s)
            | BlockSize::Read8(s)
            | BlockSize::Read16(s)
            | BlockSize::Samples256to32768(s) => s,
        }
    }
}

impl TryFrom<u16> for BlockSize<u16> {
    type Error = Error;

    fn try_from(size: u16) -> Result<Self, Error> {
        match size {
            0 => Err(Error::InvalidBlockSize),
            192 => Ok(Self::Samples192),
            576 | 1152 | 2304 | 4608 => Ok(Self::Samples576to4608(size)),
            256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 => {
                Ok(Self::Samples256to32768(size))
            }
            size if size <= 256 => Ok(Self::Read8(size)),
            size => Ok(Self::Read16(size)),
        }
    }
}

/// Possible sample rates in a FLAC frame
#[derive(Copy, Clone, Debug)]
pub enum SampleRate<R> {
    /// Get rate from STREAMINFO metadata block
    Streaminfo(R),
    /// 88200 Hz
    Hz88200,
    /// 176,400 Hz
    Hz176400,
    /// 192,000 Hz
    Hz192000,
    /// 8,000 Hz
    Hz8000,
    /// 16,000 Hz
    Hz16000,
    /// 22,050 Hz
    Hz22050,
    /// 24,000 Hz
    Hz24000,
    /// 32,000 Hz
    Hz32000,
    /// 44,100 Hz
    Hz44100,
    /// 48,000 Hz
    Hz48000,
    /// 96,000 Hz
    Hz96000,
    /// 8-bit value * 1,000 Hz
    Read8x1000(R),
    /// 16-bit value in Hz
    Read16(R),
    /// 16-bit value * 10 in Hz
    Read16x10(R),
}

/// Reads the raw sample rate bits, which need to be finalized
impl FromBitStream for SampleRate<()> {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        match r.read::<4, u8>()? {
            0b0000 => Ok(Self::Streaminfo(())),
            0b0001 => Ok(Self::Hz88200),
            0b0010 => Ok(Self::Hz176400),
            0b0011 => Ok(Self::Hz192000),
            0b0100 => Ok(Self::Hz8000),
            0b0101 => Ok(Self::Hz16000),
            0b0110 => Ok(Self::Hz22050),
            0b0111 => Ok(Self::Hz24000),
            0b1000 => Ok(Self::Hz32000),
            0b1001 => Ok(Self::Hz44100),
            0b1010 => Ok(Self::Hz48000),
            0b1011 => Ok(Self::Hz96000),
            0b1100 => Ok(Self::Read8x1000(())),
            0b1101 => Ok(Self::Read16(())),
            0b1110 => Ok(Self::Read16x10(())),
            0b1111 => Err(Error::InvalidSampleRate),
            _ => unreachable!(), // 4-bit field
        }
    }
}

impl SampleRate<()> {
    fn finalize_read<R: BitRead + ?Sized>(
        self,
        r: &mut R,
        non_subset_rate: impl FnOnce() -> Result<u32, Error>,
    ) -> Result<SampleRate<u32>, Error> {
        match self {
            Self::Streaminfo(()) => Ok(SampleRate::Streaminfo(non_subset_rate()?)),
            Self::Hz88200 => Ok(SampleRate::Hz88200),
            Self::Hz176400 => Ok(SampleRate::Hz176400),
            Self::Hz192000 => Ok(SampleRate::Hz192000),
            Self::Hz8000 => Ok(SampleRate::Hz8000),
            Self::Hz16000 => Ok(SampleRate::Hz16000),
            Self::Hz22050 => Ok(SampleRate::Hz22050),
            Self::Hz24000 => Ok(SampleRate::Hz24000),
            Self::Hz32000 => Ok(SampleRate::Hz32000),
            Self::Hz44100 => Ok(SampleRate::Hz44100),
            Self::Hz48000 => Ok(SampleRate::Hz48000),
            Self::Hz96000 => Ok(SampleRate::Hz96000),
            Self::Read8x1000(()) => Ok(SampleRate::Read8x1000(r.read::<8, u32>()? * 1000)),
            Self::Read16(()) => Ok(SampleRate::Read16(r.read::<16, _>()?)),
            Self::Read16x10(()) => Ok(SampleRate::Read16x10(r.read::<16, u32>()? * 10)),
        }
    }
}

/// Writes the raw sample rate bits
impl<R> ToBitStream for SampleRate<R> {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<4, u8>(match self {
            Self::Streaminfo(_) => 0b0000,
            Self::Hz88200 => 0b0001,
            Self::Hz176400 => 0b0010,
            Self::Hz192000 => 0b0011,
            Self::Hz8000 => 0b0100,
            Self::Hz16000 => 0b0101,
            Self::Hz22050 => 0b0110,
            Self::Hz24000 => 0b0111,
            Self::Hz32000 => 0b1000,
            Self::Hz44100 => 0b1001,
            Self::Hz48000 => 0b1010,
            Self::Hz96000 => 0b1011,
            Self::Read8x1000(_) => 0b1100,
            Self::Read16(_) => 0b1101,
            Self::Read16x10(_) => 0b1110,
        })
    }
}

impl From<SampleRate<u32>> for u32 {
    fn from(rate: SampleRate<u32>) -> Self {
        match rate {
            SampleRate::Streaminfo(u)
            | SampleRate::Read8x1000(u)
            | SampleRate::Read16(u)
            | SampleRate::Read16x10(u) => u,
            SampleRate::Hz88200 => 88200,
            SampleRate::Hz176400 => 176400,
            SampleRate::Hz192000 => 192000,
            SampleRate::Hz8000 => 8000,
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz22050 => 22050,
            SampleRate::Hz24000 => 24000,
            SampleRate::Hz32000 => 32000,
            SampleRate::Hz44100 => 44100,
            SampleRate::Hz48000 => 48000,
            SampleRate::Hz96000 => 96000,
        }
    }
}

impl TryFrom<u32> for SampleRate<u32> {
    type Error = Error;

    fn try_from(sample_rate: u32) -> Result<Self, Error> {
        match sample_rate {
            88200 => Ok(Self::Hz88200),
            176400 => Ok(Self::Hz176400),
            192000 => Ok(Self::Hz192000),
            8000 => Ok(Self::Hz8000),
            16000 => Ok(Self::Hz16000),
            22050 => Ok(Self::Hz22050),
            24000 => Ok(Self::Hz24000),
            32000 => Ok(Self::Hz32000),
            44100 => Ok(Self::Hz44100),
            48000 => Ok(Self::Hz48000),
            96000 => Ok(Self::Hz96000),
            rate if (rate % 1000) == 0 && (rate / 1000) < u8::MAX as u32 => {
                Ok(Self::Read8x1000(rate))
            }
            rate if (rate % 10) == 0 && (rate / 10) < u16::MAX as u32 => Ok(Self::Read16x10(rate)),
            rate if rate < u16::MAX as u32 => Ok(Self::Read16(rate)),
            rate if rate < 1 << 20 => Ok(Self::Streaminfo(rate)),
            _ => Err(Error::InvalidSampleRate),
        }
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

/// A frame number in the stream, as FLAC frames or samples
#[derive(Copy, Clone, Debug, Default)]
pub struct FrameNumber(pub u64);

impl FrameNumber {
    /// Attempt to increment frame number
    ///
    /// # Error
    ///
    /// Returns an error if the frame number is too large
    pub fn try_increment(&mut self) -> Result<(), Error> {
        // TODO - implement number check
        self.0 += 1;
        Ok(())
    }
}

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
                w.write::<8, _>([byte(v, 1), byte(v, 0)])?;
                Ok(())
            }
            v @ 0x1_0000..=0x1F_FFFF => {
                w.write_unary::<0>(4)?;
                w.write::<3, _>(v >> (6 * 3))?;
                w.write::<8, _>([byte(v, 2), byte(v, 1), byte(v, 0)])?;
                Ok(())
            }
            v @ 0x20_0000..=0x3FF_FFFF => {
                w.write_unary::<0>(5)?;
                w.write::<2, _>(v >> (6 * 4))?;
                w.write::<8, _>([byte(v, 3), byte(v, 2), byte(v, 1), byte(v, 0)])?;
                Ok(())
            }
            v @ 0x400_0000..=0x7FFF_FFFF => {
                w.write_unary::<0>(6)?;
                w.write::<1, _>(v >> (6 * 5))?;
                w.write::<8, _>([byte(v, 4), byte(v, 3), byte(v, 2), byte(v, 1), byte(v, 0)])?;
                Ok(())
            }
            v @ 0x8000_0000..=0xF_FFFF_FFFF => {
                w.write_unary::<0>(7)?;
                w.write::<8, _>([
                    byte(v, 5),
                    byte(v, 4),
                    byte(v, 3),
                    byte(v, 2),
                    byte(v, 1),
                    byte(v, 0),
                ])?;
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
    /// The number of wasted bits-per-sample
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
                w.write_unary::<1>(wasted)?;
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
    Fixed {
        /// The predictor order, from 0..5
        order: u8,
    },
    /// Samples are stored with dynamic LPC parameters
    Lpc {
        /// The predictor order, from 1..33
        order: NonZero<u8>,
    },
}

impl SubframeHeaderType {
    /// A set of FIXED subframe coefficients
    ///
    /// Note that these are in the reverse order from how
    /// they're usually presented, simply because we'll
    /// be predicting samples in reverse order.
    pub const FIXED_COEFFS: [&[i64]; 5] = [&[], &[1], &[2, -1], &[3, -3, 1], &[4, -6, 4, -1]];
}

impl FromBitStream for SubframeHeaderType {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read::<6, u8>()? {
            0b000000 => Ok(Self::Constant),
            0b000001 => Ok(Self::Verbatim),
            v @ 0b001000..=0b001100 => Ok(Self::Fixed {
                order: v - 0b001000,
            }),
            v @ 0b100000..=0b111111 => Ok(Self::Lpc {
                order: NonZero::new(v - 31).unwrap(),
            }),
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
            Self::Fixed { order } => 0b001000 + order,
            Self::Lpc { order } => order.get() + 31,
        })?;
        Ok(())
    }
}

/// A FLAC residual partition header
#[derive(Debug)]
pub enum ResidualPartitionHeader<const RICE_MAX: u32> {
    /// Standard, un-escaped partition
    Standard {
        /// The partition's Rice parameter
        rice: BitCount<RICE_MAX>,
    },
    /// Escaped partition
    Escaped {
        /// The size of each residual in bits
        escape_size: SignedBitCount<0b11111>,
    },
    /// All residuals in partition are 0
    Constant,
}

impl<const RICE_MAX: u32> FromBitStream for ResidualPartitionHeader<RICE_MAX> {
    type Error = std::io::Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let rice = r.read_count()?;

        if rice == BitCount::new::<{ RICE_MAX }>() {
            match r.read_count()?.signed_count() {
                Some(escape_size) => Ok(Self::Escaped { escape_size }),
                None => Ok(Self::Constant),
            }
        } else {
            Ok(Self::Standard { rice })
        }
    }
}

impl<const RICE_MAX: u32> ToBitStream for ResidualPartitionHeader<RICE_MAX> {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        match self {
            Self::Standard { rice } => w.write_count(*rice),
            Self::Escaped { escape_size } => {
                w.write_count(BitCount::<RICE_MAX>::new::<{ RICE_MAX }>())?;
                w.write_count(escape_size.count())
            }
            Self::Constant => {
                w.write_count(BitCount::<RICE_MAX>::new::<{ RICE_MAX }>())?;
                w.write_count(BitCount::<0b11111>::new::<0>())
            }
        }
    }
}

/// A whole FLAC frame
#[derive(Debug)]
pub struct Frame {
    /// The FLAC frame's header
    pub header: FrameHeader,
    /// A FLAC frame's sub-frames
    pub subframes: Vec<Subframe>,
}

impl Frame {
    /// Reads new frame from the given reader
    pub fn read<R: std::io::Read>(reader: &mut R, streaminfo: &Streaminfo) -> Result<Self, Error> {
        use crate::crc::{Checksum, Crc16, CrcReader};
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc16_reader: CrcReader<_, Crc16> = CrcReader::new(reader.by_ref());

        let header = FrameHeader::read(crc16_reader.by_ref(), streaminfo)?;

        let mut reader = BitReader::endian(crc16_reader.by_ref(), BigEndian);

        let subframes = match header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => (0..total_channels)
                .map(|_| {
                    reader
                        .parse_with::<Subframe>(&(header.block_size.into(), header.bits_per_sample))
                })
                .collect::<Result<Vec<_>, _>>()?,
            ChannelAssignment::LeftSide => vec![
                reader.parse_with(&(header.block_size.into(), header.bits_per_sample))?,
                reader.parse_with(&(
                    header.block_size.into(),
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                ))?,
            ],
            ChannelAssignment::SideRight => vec![
                reader.parse_with(&(
                    header.block_size.into(),
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                ))?,
                reader.parse_with(&(header.block_size.into(), header.bits_per_sample))?,
            ],
            ChannelAssignment::MidSide => vec![
                reader.parse_with(&(header.block_size.into(), header.bits_per_sample))?,
                reader.parse_with(&(
                    header.block_size.into(),
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                ))?,
            ],
        };

        reader.byte_align();
        reader.skip(16)?; // CRC-16 checksum

        if crc16_reader.into_checksum().valid() {
            Ok(Self { header, subframes })
        } else {
            Err(Error::Crc16Mismatch)
        }
    }

    /// Writes frame to the given writer
    pub fn write<W: std::io::Write>(
        &self,
        streaminfo: &Streaminfo,
        writer: &mut W,
    ) -> Result<(), Error> {
        use crate::crc::{Crc16, CrcWriter};
        use bitstream_io::{BigEndian, BitWriter};
        use std::io::Write;

        let mut crc16_writer: CrcWriter<_, Crc16> = CrcWriter::new(writer.by_ref());

        self.header.write(crc16_writer.by_ref(), streaminfo)?;

        let mut writer = BitWriter::endian(crc16_writer.by_ref(), BigEndian);

        match self.header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => {
                assert_eq!(total_channels as usize, self.subframes.len());

                for subframe in &self.subframes {
                    writer.build_with(subframe, &self.header.bits_per_sample)?;
                }
            }
            ChannelAssignment::LeftSide => match self.subframes.as_slice() {
                [left, side] => {
                    writer.build_with(left, &self.header.bits_per_sample)?;

                    writer.build_with(
                        side,
                        &self
                            .header
                            .bits_per_sample
                            .checked_add(1)
                            .ok_or(Error::ExcessiveBps)?,
                    )?;
                }
                _ => panic!("incorrect subframe count for left-side"),
            },
            ChannelAssignment::SideRight => match self.subframes.as_slice() {
                [side, right] => {
                    writer.build_with(
                        side,
                        &self
                            .header
                            .bits_per_sample
                            .checked_add(1)
                            .ok_or(Error::ExcessiveBps)?,
                    )?;

                    writer.build_with(right, &self.header.bits_per_sample)?;
                }
                _ => panic!("incorrect subframe count for left-side"),
            },
            ChannelAssignment::MidSide => match self.subframes.as_slice() {
                [mid, side] => {
                    writer.build_with(mid, &self.header.bits_per_sample)?;

                    writer.build_with(
                        side,
                        &self
                            .header
                            .bits_per_sample
                            .checked_add(1)
                            .ok_or(Error::ExcessiveBps)?,
                    )?;
                }
                _ => panic!("incorrect subframe count for left-side"),
            },
        }

        let crc16: u16 = writer.aligned_writer()?.checksum().into();
        writer.write_from(crc16)?;

        Ok(())
    }
}

/// A FLAC's frame's subframe, one per channel
#[derive(Debug)]
pub enum Subframe {
    /// A CONSTANT subframe, in which all samples are identical
    Constant {
        /// The subframe's sample
        sample: i32,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
    /// A VERBATIM subframe, in which all samples are stored uncompressed
    Verbatim {
        /// The subframe's samples
        samples: Vec<i32>,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
    /// A FIXED subframe, encoded with a fixed set of parameters
    Fixed {
        /// The subframe's predictor order from 0 to 4 (inclusive)
        order: u8,
        /// The subframe's warm-up samples (one per order)
        warm_up: Vec<i32>,
        /// The subframe's residuals
        residuals: Residuals,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
    /// An LPC subframe, encoded with a variable set of parameters
    Lpc {
        /// The subframe's predictor order
        order: NonZero<u8>,
        /// The subframe's warm-up samples (one per order)
        warm_up: Vec<i32>,
        /// The subframe's QLP precision
        precision: SignedBitCount<15>,
        /// The subframe's QLP shift
        shift: u32,
        /// The subframe's QLP coefficients (one per order)
        coefficients: Vec<i32>,
        /// The subframe's residuals
        residuals: Residuals,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
}

impl FromBitStreamWith<'_> for Subframe {
    type Context = (u16, SignedBitCount<32>);
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        (block_size, bits_per_sample): &(u16, SignedBitCount<32>),
    ) -> Result<Self, Error> {
        match r.parse()? {
            SubframeHeader {
                type_: SubframeHeaderType::Constant,
                wasted_bps,
            } => Ok(Self::Constant {
                sample: r.read_signed_counted(
                    bits_per_sample
                        .checked_sub::<32>(wasted_bps)
                        .ok_or(Error::ExcessiveWastedBits)?,
                )?,
                wasted_bps,
            }),
            SubframeHeader {
                type_: SubframeHeaderType::Verbatim,
                wasted_bps,
            } => {
                let effective_bps = bits_per_sample
                    .checked_sub::<32>(wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?;

                Ok(Self::Verbatim {
                    samples: (0..*block_size)
                        .map(|_| r.read_signed_counted::<32, i32>(effective_bps))
                        .collect::<Result<Vec<_>, _>>()?,
                    wasted_bps,
                })
            }
            SubframeHeader {
                type_: SubframeHeaderType::Fixed { order },
                wasted_bps,
            } => {
                let effective_bps = bits_per_sample
                    .checked_sub::<32>(wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?;

                Ok(Self::Fixed {
                    order,
                    warm_up: (0..order)
                        .map(|_| r.read_signed_counted::<32, i32>(effective_bps))
                        .collect::<Result<Vec<_>, _>>()?,
                    residuals: r.parse_with(&((*block_size).into(), order.into()))?,
                    wasted_bps,
                })
            }
            SubframeHeader {
                type_: SubframeHeaderType::Lpc { order },
                wasted_bps,
            } => {
                let effective_bps = bits_per_sample
                    .checked_sub::<32>(wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?;

                let warm_up = (0..order.get())
                    .map(|_| r.read_signed_counted::<32, i32>(effective_bps))
                    .collect::<Result<Vec<_>, _>>()?;

                let precision: SignedBitCount<15> = r
                    .read_count::<0b1111>()?
                    .checked_add(1)
                    .and_then(|c| c.signed_count())
                    .ok_or(Error::InvalidQlpPrecision)?;

                let shift: u32 = r
                    .read::<5, i32>()?
                    .try_into()
                    .map_err(|_| Error::NegativeLpcShift)?;

                let coefficients = (0..order.get())
                    .map(|_| r.read_signed_counted(precision))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(Self::Lpc {
                    order,
                    warm_up,
                    precision,
                    shift,
                    coefficients,
                    residuals: r.parse_with(&((*block_size).into(), order.get().into()))?,
                    wasted_bps,
                })
            }
        }
    }
}

impl ToBitStreamWith<'_> for Subframe {
    type Context = SignedBitCount<32>;
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        bits_per_sample: &SignedBitCount<32>,
    ) -> Result<(), Error> {
        match self {
            Self::Constant { sample, wasted_bps } => {
                w.build(&SubframeHeader {
                    type_: SubframeHeaderType::Constant,
                    wasted_bps: *wasted_bps,
                })?;

                w.write_signed_counted(
                    bits_per_sample
                        .checked_sub::<32>(*wasted_bps)
                        .ok_or(Error::ExcessiveWastedBits)?,
                    *sample,
                )?;

                Ok(())
            }
            Self::Verbatim {
                samples,
                wasted_bps,
            } => {
                let effective_bps = bits_per_sample
                    .checked_sub::<32>(*wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?;

                w.build(&SubframeHeader {
                    type_: SubframeHeaderType::Verbatim,
                    wasted_bps: *wasted_bps,
                })?;

                for sample in samples {
                    w.write_signed_counted(effective_bps, *sample)?;
                }

                Ok(())
            }
            Self::Fixed {
                order,
                warm_up,
                residuals,
                wasted_bps,
            } => {
                assert_eq!(*order as usize, warm_up.len());

                let effective_bps = bits_per_sample
                    .checked_sub::<32>(*wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?;

                w.build(&SubframeHeader {
                    type_: SubframeHeaderType::Fixed { order: *order },
                    wasted_bps: *wasted_bps,
                })?;

                for sample in warm_up {
                    w.write_signed_counted(effective_bps, *sample)?;
                }

                w.build(residuals)?;

                Ok(())
            }
            Self::Lpc {
                order,
                warm_up,
                precision,
                shift,
                coefficients,
                residuals,
                wasted_bps,
            } => {
                assert_eq!(order.get() as usize, warm_up.len());
                assert_eq!(order.get() as usize, coefficients.len());

                let effective_bps = bits_per_sample
                    .checked_sub::<32>(*wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?;

                w.build(&SubframeHeader {
                    type_: SubframeHeaderType::Lpc { order: *order },
                    wasted_bps: *wasted_bps,
                })?;

                for sample in warm_up {
                    w.write_signed_counted(effective_bps, *sample)?;
                }

                w.write_count(
                    precision
                        .checked_sub::<0b1111>(1)
                        .ok_or(Error::InvalidQlpPrecision)?
                        .count(),
                )?;

                w.write::<5, i32>(i32::try_from(*shift).unwrap())?;

                for coeff in coefficients {
                    w.write_signed_counted(*precision, *coeff)?;
                }

                w.build(residuals)?;

                Ok(())
            }
        }
    }
}

/// Residual values for FIXED or LPC subframes
#[derive(Debug)]
pub enum Residuals {
    /// Coding method 0
    Method0 {
        /// The residual partitions
        partitions: Vec<ResidualPartition<0b1111>>,
    },
    /// Coding method 1
    Method1 {
        /// The residual partitions
        partitions: Vec<ResidualPartition<0b11111>>,
    },
}

impl FromBitStreamWith<'_> for Residuals {
    type Context = (usize, usize);
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, params: &(usize, usize)) -> Result<Self, Error> {
        fn read_partitions<const RICE_MAX: u32, R: BitRead + ?Sized>(
            reader: &mut R,
            (block_size, predictor_order): &(usize, usize),
        ) -> Result<Vec<ResidualPartition<RICE_MAX>>, Error> {
            let partition_order = reader.read::<4, u32>()?;
            let partition_count = 1 << partition_order;

            (0..partition_count)
                .map(|p| {
                    let partition_size = (block_size / partition_count)
                        .checked_sub(if p == 0 { *predictor_order } else { 0 })
                        .ok_or(Error::InvalidPartitionOrder)?;

                    reader.parse_with(&partition_size)
                })
                .collect()
        }

        match r.read::<2, u8>()? {
            0 => Ok(Self::Method0 {
                partitions: read_partitions::<0b1111, R>(r, params)?,
            }),
            1 => Ok(Self::Method1 {
                partitions: read_partitions::<0b11111, R>(r, params)?,
            }),
            _ => Err(Error::InvalidCodingMethod),
        }
    }
}

impl ToBitStream for Residuals {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        fn write_partitions<const RICE_MAX: u32, W: BitWrite + ?Sized>(
            writer: &mut W,
            partitions: &[ResidualPartition<RICE_MAX>],
        ) -> Result<(), Error> {
            assert!(!partitions.is_empty());
            assert!(partitions.len().is_power_of_two());

            writer.write::<4, _>(partitions.len().ilog2())?;

            for partition in partitions.iter() {
                writer.build(partition)?;
            }

            Ok(())
        }

        match self {
            Self::Method0 { partitions } => {
                w.write::<2, u8>(0)?; // coding method
                write_partitions(w, partitions)
            }
            Self::Method1 { partitions } => {
                w.write::<2, u8>(1)?; // coding method
                write_partitions(w, partitions)
            }
        }
    }
}

/// An individual residual block partition
#[derive(Debug)]
pub enum ResidualPartition<const RICE_MAX: u32> {
    /// A standard residual partition
    Standard {
        /// The partition's Rice parameter
        rice: BitCount<RICE_MAX>,
        /// The partition's residuals
        residuals: Vec<i32>,
    },
    /// An escaped residual partition
    Escaped {
        /// The size of each residual in bits
        escape_size: SignedBitCount<0b11111>,
        /// The partition's residuals
        residuals: Vec<i32>,
    },
    /// A partition in which all residuals are 0
    Constant,
}

impl<const RICE_MAX: u32> FromBitStreamWith<'_> for ResidualPartition<RICE_MAX> {
    type Context = usize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, partition_len: &usize) -> Result<Self, Error> {
        match r.parse::<ResidualPartitionHeader<RICE_MAX>>()? {
            ResidualPartitionHeader::Standard { rice } => Ok(Self::Standard {
                residuals: (0..*partition_len)
                    .map(|_| {
                        let msb = r.read_unary::<1>()?;
                        let lsb = r.read_counted::<RICE_MAX, u32>(rice)?;
                        let unsigned = msb << u32::from(rice) | lsb;
                        Ok::<_, Error>(if (unsigned & 1) == 1 {
                            -((unsigned >> 1) as i32) - 1
                        } else {
                            (unsigned >> 1) as i32
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                rice,
            }),
            ResidualPartitionHeader::Escaped { escape_size } => Ok(Self::Escaped {
                residuals: (0..*partition_len)
                    .map(|_| r.read_signed_counted(escape_size))
                    .collect::<Result<Vec<_>, _>>()?,
                escape_size,
            }),
            ResidualPartitionHeader::Constant => Ok(Self::Constant),
        }
    }
}

impl<const RICE_MAX: u32> ToBitStream for ResidualPartition<RICE_MAX> {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        match self {
            Self::Standard { residuals, rice } => {
                w.build(&ResidualPartitionHeader::Standard { rice: *rice })?;

                let shift = 1 << u32::from(*rice);

                for residual in residuals {
                    let unsigned = if residual.is_negative() {
                        ((-*residual as u32 - 1) << 1) + 1
                    } else {
                        (*residual as u32) << 1
                    };
                    let (quot, rem) = (unsigned / shift, unsigned % shift);
                    w.write_unary::<1>(quot)?;
                    w.write_counted(*rice, rem)?;
                }
                Ok(())
            }
            Self::Escaped {
                escape_size,
                residuals,
            } => {
                w.build(&ResidualPartitionHeader::<RICE_MAX>::Escaped {
                    escape_size: *escape_size,
                })?;

                for residual in residuals {
                    w.write_signed_counted(*escape_size, *residual)?;
                }

                Ok(())
            }
            Self::Constant => Ok(w.build(&ResidualPartitionHeader::<RICE_MAX>::Constant)?),
        }
    }
}
