// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For handling common FLAC stream items

use crate::Error;
use crate::crc::{Checksum, Crc16, CrcReader, CrcWriter};
use crate::metadata::Streaminfo;
use bitstream_io::{
    BitCount, BitRead, BitWrite, FromBitStream, FromBitStreamUsing, FromBitStreamWith,
    SignedBitCount, ToBitStream, ToBitStreamUsing, ToBitStreamWith,
};
use std::num::NonZero;

/// A common trait for signed integers
pub trait SignedInteger:
    bitstream_io::SignedInteger
    + std::ops::Shl<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + Into<i64>
{
    /// Unconditionally converts a u32 to ourself
    fn from_u32(u: u32) -> Self;

    /// Unconditionally converts ourself to u32
    fn to_u32(self) -> u32;

    /// Unconditionally converts i64 to ourself
    fn from_i64(i: i64) -> Self;
}

impl SignedInteger for i32 {
    #[inline]
    fn from_u32(u: u32) -> Self {
        u as i32
    }

    #[inline]
    fn to_u32(self) -> u32 {
        self as u32
    }

    fn from_i64(i: i64) -> Self {
        i as i32
    }
}

impl SignedInteger for i64 {
    #[inline]
    fn from_u32(u: u32) -> Self {
        u as i64
    }

    #[inline]
    fn to_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    fn from_i64(i: i64) -> Self {
        i
    }
}

/// A FLAC frame header
///
/// | Bits      | Field |
/// |----------:|-------|
/// | 15        | sync code (`0b111111111111100`)
/// | 1         | `blocking_strategy`
/// | 4         | `block_size`
/// | 4         | `sample_rate`
/// | 4         | `channel_assignment`
/// | 3         | `bits_per_sample`
/// | 1         | padding (0)
/// | 8-56      | `frame_number`
/// | (8 or 16) | uncommon block size
/// | (8 or 16) | uncommon sample rate
/// | 8         | CRC-8
///
/// # Example
/// ```
/// use flac_codec::stream::{
///     FrameHeader, BlockSize, SampleRate, ChannelAssignment,
///     Independent, BitsPerSample, FrameNumber,
/// };
///
/// let mut data: &[u8] = &[
///     0b11111111, 0b1111100_0,  // sync code + blocking
///     0b0110_1001,              // block size + sample rate
///     0b0000_100_0,             // channels + bps + pad
///     0x00,                     // frame number
///     0x13,                     // uncommon block size (+1)
///     0x64,                     // CRC-8
/// ];
///
/// assert_eq!(
///     FrameHeader::read_subset(&mut data).unwrap(),
///     FrameHeader {
///         blocking_strategy: false,
///         block_size: BlockSize::Uncommon8(20),
///         sample_rate: SampleRate::Hz44100,
///         channel_assignment: ChannelAssignment::Independent(
///             Independent::Mono
///         ),
///         bits_per_sample: BitsPerSample::Bps16,
///         frame_number: FrameNumber(0),
///     },
/// );
/// ```
#[derive(Debug, Eq, PartialEq)]
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
    // pub bits_per_sample: SignedBitCount<32>,
    pub bits_per_sample: BitsPerSample,
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

    /// Reads new header from the given reader
    pub fn read_subset<R: std::io::Read>(reader: &mut R) -> Result<Self, Error> {
        use crate::crc::{Checksum, Crc8, CrcReader};
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc8: CrcReader<_, Crc8> = CrcReader::new(reader);
        BitReader::endian(crc8.by_ref(), BigEndian)
            .parse()
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

    /// Builds header to the given writer
    pub fn write_subset<W: std::io::Write>(&self, writer: &mut W) -> Result<(), Error> {
        use crate::crc::{Crc8, CrcWriter};
        use bitstream_io::{BigEndian, BitWriter};
        use std::io::Write;

        let mut crc8: CrcWriter<_, Crc8> = CrcWriter::new(writer.by_ref());
        BitWriter::endian(crc8.by_ref(), BigEndian).build(self)?;
        let crc8 = crc8.into_checksum().into();
        writer.write_all(std::slice::from_ref(&crc8))?;
        Ok(())
    }

    fn parse<R: BitRead + ?Sized>(
        r: &mut R,
        non_subset_rate: Option<u32>,
        non_subset_bps: Option<SignedBitCount<32>>,
    ) -> Result<Self, Error> {
        r.read_const::<15, { Self::SYNC_CODE }, _>(Error::InvalidSyncCode)?;
        let blocking_strategy = r.read_bit()?;
        let encoded_block_size = r.parse()?;
        let encoded_sample_rate = r.parse_using(non_subset_rate)?;
        let channel_assignment = r.parse()?;
        let bits_per_sample = r.parse_using(non_subset_bps)?;
        r.skip(1)?;
        let frame_number = r.parse()?;

        let frame_header = Self {
            blocking_strategy,
            frame_number,
            block_size: r.parse_using(encoded_block_size)?,
            sample_rate: r.parse_using(encoded_sample_rate)?,
            channel_assignment,
            bits_per_sample,
        };

        r.skip(8)?; // CRC-8

        Ok(frame_header)
    }

    fn build<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        w.write_const::<15, { Self::SYNC_CODE }>()?;
        w.write_bit(self.blocking_strategy)?;
        w.build(&self.block_size)?;
        w.build(&self.sample_rate)?;
        w.build(&self.channel_assignment)?;
        w.build(&self.bits_per_sample)?;
        w.pad(1)?;
        w.build(&self.frame_number)?;

        // uncommon block size
        match self.block_size {
            BlockSize::Uncommon8(size) => {
                w.write::<8, _>(size.checked_sub(1).ok_or(Error::InvalidBlockSize)?)?
            }
            BlockSize::Uncommon16(size) => {
                w.write::<16, _>(size.checked_sub(1).ok_or(Error::InvalidBlockSize)?)?
            }
            _ => { /* do nothing */ }
        }

        // uncommon sample rate
        match self.sample_rate {
            SampleRate::KHz(rate) => w.write::<8, _>(rate / 1000)?,
            SampleRate::Hz(rate) => {
                w.write::<16, _>(rate)?;
            }
            SampleRate::DHz(rate) => {
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
            Some(streaminfo.sample_rate),
            Some(streaminfo.bits_per_sample),
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

impl FromBitStream for FrameHeader {
    type Error = Error;

    #[inline]
    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        FrameHeader::parse(r, None, None)
    }
}

impl ToBitStreamWith<'_> for FrameHeader {
    type Error = Error;
    type Context = Streaminfo;

    #[inline]
    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        _streaminfo: &Streaminfo,
    ) -> Result<(), Self::Error> {
        self.build(w)
    }
}

impl ToBitStream for FrameHeader {
    type Error = Error;

    #[inline]
    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        self.build(w)
    }
}

/// Possible block sizes in a FLAC frame, in samples
///
/// Common sizes are stored as a 4-bit value,
/// while uncommon sizes are stored as 8 or 16 bit values.
///
/// | Bits   | Block Size |
/// |-------:|------------|
/// | `0000` | invalid
/// | `0001` | 192
/// | `0010` | 576
/// | `0011` | 1152
/// | `0100` | 2304
/// | `0101` | 4606
/// | `0110` | 8 bit field (+1)
/// | `0111` | 16 bit field (+1)
/// | `1000` | 256
/// | `1001` | 512
/// | `1010` | 1024
/// | `1011` | 2048
/// | `1100` | 4096
/// | `1101` | 8192
/// | `1110` | 16384
/// | `1111` | 32768
///
/// Handing uncommon block sizes is why this type
/// is a generic with two different implementations
/// from reading from a bitstream.
/// This first reads common sizes, while the second
/// reads additional bits if necessary.
///
/// # Example
///
/// ```
/// use flac_codec::stream::BlockSize;
/// use bitstream_io::{BitReader, BitRead, BigEndian};
///
/// let data: &[u8] = &[
///     0b0110_1001,              // block size + sample rate
///     0b0000_100_0,             // channels + bps + pad
///     0x00,                     // frame number
///     0x13,                     // uncommon block size (+1)
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// let block_size = r.parse::<BlockSize<()>>().unwrap();  // reads 0b0110
/// assert_eq!(
///     block_size,
///     BlockSize::Uncommon8(()),  // need to read actual block size from end of frame
/// );
/// r.skip(4 + 8 + 8).unwrap();    // skip unnecessary bits for this example
/// assert_eq!(
///     // read remainder of block size from end of frame
///     r.parse_using::<BlockSize<u16>>(block_size).unwrap(),
///     BlockSize::Uncommon8(0x13 + 1),
/// );
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockSize<B> {
    /// 192 samples
    Samples192,
    /// 576 samples
    Samples576,
    /// 1152 samples
    Samples1152,
    /// 2304 samples
    Samples2304,
    /// 4608 samples
    Samples4608,
    /// Uncommon 8 bit sample count + 1
    Uncommon8(B),
    /// Uncommon 16 bit sample count + 1
    Uncommon16(B),
    /// 256 samples
    Samples256,
    /// 512 samples
    Samples512,
    /// 1024 samples
    Samples1024,
    /// 2048 samples
    Samples2048,
    /// 4096 samples
    Samples4096,
    /// 8192 samples
    Samples8192,
    /// 16384 samples
    Samples16384,
    /// 32768 samples
    Samples32768,
}

impl FromBitStream for BlockSize<()> {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        match r.read::<4, u8>()? {
            0b0000 => Err(Error::InvalidBlockSize),
            0b0001 => Ok(Self::Samples192),
            0b0010 => Ok(Self::Samples576),
            0b0011 => Ok(Self::Samples1152),
            0b0100 => Ok(Self::Samples2304),
            0b0101 => Ok(Self::Samples4608),
            0b0110 => Ok(Self::Uncommon8(())),
            0b0111 => Ok(Self::Uncommon16(())),
            0b1000 => Ok(Self::Samples256),
            0b1001 => Ok(Self::Samples512),
            0b1010 => Ok(Self::Samples1024),
            0b1011 => Ok(Self::Samples2048),
            0b1100 => Ok(Self::Samples4096),
            0b1101 => Ok(Self::Samples8192),
            0b1110 => Ok(Self::Samples16384),
            0b1111 => Ok(Self::Samples32768),
            0b10000.. => unreachable!(), // 4-bit field
        }
    }
}

impl FromBitStreamUsing for BlockSize<u16> {
    type Context = BlockSize<()>;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: BlockSize<()>) -> Result<Self, Error> {
        match size {
            BlockSize::Samples192 => Ok(Self::Samples192),
            BlockSize::Samples576 => Ok(Self::Samples576),
            BlockSize::Samples1152 => Ok(Self::Samples1152),
            BlockSize::Samples2304 => Ok(Self::Samples2304),
            BlockSize::Samples4608 => Ok(Self::Samples4608),
            BlockSize::Samples256 => Ok(Self::Samples256),
            BlockSize::Samples512 => Ok(Self::Samples512),
            BlockSize::Samples1024 => Ok(Self::Samples1024),
            BlockSize::Samples2048 => Ok(Self::Samples2048),
            BlockSize::Samples4096 => Ok(Self::Samples4096),
            BlockSize::Samples8192 => Ok(Self::Samples8192),
            BlockSize::Samples16384 => Ok(Self::Samples16384),
            BlockSize::Samples32768 => Ok(Self::Samples32768),
            BlockSize::Uncommon8(()) => Ok(Self::Uncommon8(r.read::<8, u16>()? + 1)),
            BlockSize::Uncommon16(()) => Ok(Self::Uncommon16(
                r.read::<16, u16>()?
                    .checked_add(1)
                    .ok_or(Error::InvalidBlockSize)?,
            )),
        }
    }
}

impl<B> ToBitStream for BlockSize<B> {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<4, u8>(match self {
            Self::Samples192 => 0b0001,
            Self::Samples576 => 0b0010,
            Self::Samples1152 => 0b0011,
            Self::Samples2304 => 0b0100,
            Self::Samples4608 => 0b0101,
            Self::Uncommon8(_) => 0b0110,
            Self::Uncommon16(_) => 0b0111,
            Self::Samples256 => 0b1000,
            Self::Samples512 => 0b1001,
            Self::Samples1024 => 0b1010,
            Self::Samples2048 => 0b1011,
            Self::Samples4096 => 0b1100,
            Self::Samples8192 => 0b1101,
            Self::Samples16384 => 0b1110,
            Self::Samples32768 => 0b1111,
        })
    }
}

impl From<BlockSize<u16>> for u16 {
    fn from(size: BlockSize<u16>) -> Self {
        match size {
            BlockSize::Samples192 => 192,
            BlockSize::Samples576 => 576,
            BlockSize::Samples1152 => 1152,
            BlockSize::Samples2304 => 2304,
            BlockSize::Samples4608 => 4608,
            BlockSize::Samples256 => 256,
            BlockSize::Samples512 => 512,
            BlockSize::Samples1024 => 1024,
            BlockSize::Samples2048 => 2048,
            BlockSize::Samples4096 => 4096,
            BlockSize::Samples8192 => 8192,
            BlockSize::Samples16384 => 16384,
            BlockSize::Samples32768 => 32768,
            BlockSize::Uncommon8(s) | BlockSize::Uncommon16(s) => s,
        }
    }
}

impl TryFrom<u16> for BlockSize<u16> {
    type Error = Error;

    fn try_from(size: u16) -> Result<Self, Error> {
        match size {
            0 => Err(Error::InvalidBlockSize),
            192 => Ok(Self::Samples192),
            576 => Ok(Self::Samples576),
            1152 => Ok(Self::Samples1152),
            2304 => Ok(Self::Samples2304),
            4608 => Ok(Self::Samples4608),
            256 => Ok(Self::Samples256),
            512 => Ok(Self::Samples512),
            1024 => Ok(Self::Samples1024),
            2048 => Ok(Self::Samples2048),
            4096 => Ok(Self::Samples4096),
            8192 => Ok(Self::Samples8192),
            16384 => Ok(Self::Samples16384),
            32768 => Ok(Self::Samples32768),
            size if size <= 256 => Ok(Self::Uncommon8(size)),
            size => Ok(Self::Uncommon16(size)),
        }
    }
}

/// Possible sample rates in a FLAC frame
///
/// Common rates are stored as a 4-bit value,
/// while uncommon rates are stored as 8 or 16 bit values.
/// Sample rates defined in the STREAMINFO metadata block
/// are only possible on a "non-subset" stream, which is
/// not streamable.
///
/// | Bits   | Sample Rate |
/// |-------:|-------------|
/// | `0000` | get from STREAMINFO
/// | `0001` | 88200 Hz
/// | `0010` | 176400 Hz
/// | `0011` | 192000 Hz
/// | `0100` | 8000 Hz
/// | `0101` | 16000 Hz
/// | `0110` | 22050 Hz
/// | `0111` | 24000 Hz
/// | `1000` | 32000 Hz
/// | `1001` | 44100 Hz
/// | `1010` | 48000 Hz
/// | `1011` | 96000 Hz
/// | `1100` | read 8 bits, in kHz
/// | `1101` | read 16 bits, in Hz
/// | `1110` | read 16 bits, in 10s of Hz
/// | `1111` | invalid sample rate
///
/// Handing uncommon frame rates is why this type is a generic
/// with multiple implementations from reading from a bitstream.
/// This first reads common rates, while the second reads additional bits if necessary.
///
/// # Example
///
/// ```
/// use flac_codec::stream::SampleRate;
/// use bitstream_io::{BitReader, BitRead, BigEndian};
///
/// let data: &[u8] = &[
///     0b0110_1001,              // block size + sample rate
///     0b0000_100_0,             // channels + bps + pad
///     0x00,                     // frame number
///     0x13,                     // uncommon block size (+1)
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// r.skip(4).unwrap();          // skip block size
/// let sample_rate = r.parse::<SampleRate<()>>().unwrap();  // reads 0b1001
/// assert_eq!(
///     sample_rate,
///     SampleRate::Hz44100,     // got defined sample rate
/// );
/// r.skip(8 + 8 + 8).unwrap();  // skip unnecessary bits for this example
/// assert_eq!(
///     // since our rate is defined, no need to read additional bits
///     r.parse_using::<SampleRate<u32>>(sample_rate).unwrap(),
///     SampleRate::Hz44100,
/// );
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SampleRate<R> {
    /// Get rate from STREAMINFO metadata block
    Streaminfo(u32),
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
    /// 8-bit value in kHz
    KHz(R),
    /// 16-bit value in Hz
    Hz(R),
    /// 16-bit value * 10 in Hz
    DHz(R),
}

/// Reads the raw sample rate bits, which need to be finalized
impl FromBitStreamUsing for SampleRate<()> {
    type Context = Option<u32>;

    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        streaminfo_rate: Option<u32>,
    ) -> Result<Self, Self::Error> {
        match r.read::<4, u8>()? {
            0b0000 => Ok(Self::Streaminfo(
                streaminfo_rate.ok_or(Error::NonSubsetSampleRate)?,
            )),
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
            0b1100 => Ok(Self::KHz(())),
            0b1101 => Ok(Self::Hz(())),
            0b1110 => Ok(Self::DHz(())),
            0b1111 => Err(Error::InvalidSampleRate),
            _ => unreachable!(), // 4-bit field
        }
    }
}

impl FromBitStream for SampleRate<()> {
    type Error = Error;

    #[inline]
    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        r.parse_using(None)
    }
}

impl FromBitStreamUsing for SampleRate<u32> {
    type Context = SampleRate<()>;

    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        rate: SampleRate<()>,
    ) -> Result<Self, Self::Error> {
        match rate {
            SampleRate::Streaminfo(s) => Ok(Self::Streaminfo(s)),
            SampleRate::Hz88200 => Ok(Self::Hz88200),
            SampleRate::Hz176400 => Ok(Self::Hz176400),
            SampleRate::Hz192000 => Ok(Self::Hz192000),
            SampleRate::Hz8000 => Ok(Self::Hz8000),
            SampleRate::Hz16000 => Ok(Self::Hz16000),
            SampleRate::Hz22050 => Ok(Self::Hz22050),
            SampleRate::Hz24000 => Ok(Self::Hz24000),
            SampleRate::Hz32000 => Ok(Self::Hz32000),
            SampleRate::Hz44100 => Ok(Self::Hz44100),
            SampleRate::Hz48000 => Ok(Self::Hz48000),
            SampleRate::Hz96000 => Ok(Self::Hz96000),
            SampleRate::KHz(()) => Ok(Self::KHz(r.read::<8, u32>()? * 1000)),
            SampleRate::Hz(()) => Ok(Self::Hz(r.read::<16, _>()?)),
            SampleRate::DHz(()) => Ok(Self::DHz(r.read::<16, u32>()? * 10)),
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
            Self::KHz(_) => 0b1100,
            Self::Hz(_) => 0b1101,
            Self::DHz(_) => 0b1110,
        })
    }
}

impl From<SampleRate<u32>> for u32 {
    fn from(rate: SampleRate<u32>) -> Self {
        match rate {
            SampleRate::Streaminfo(u)
            | SampleRate::KHz(u)
            | SampleRate::Hz(u)
            | SampleRate::DHz(u) => u,
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
            rate if (rate % 1000) == 0 && (rate / 1000) < u8::MAX as u32 => Ok(Self::KHz(rate)),
            rate if (rate % 10) == 0 && (rate / 10) < u16::MAX as u32 => Ok(Self::DHz(rate)),
            rate if rate < u16::MAX as u32 => Ok(Self::Hz(rate)),
            rate if rate < 1 << 20 => Ok(Self::Streaminfo(rate)),
            _ => Err(Error::InvalidSampleRate),
        }
    }
}

/// How independent channels are stored
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Independent {
    /// 1 monoaural channel
    Mono = 1,
    /// left, right channels
    Stereo = 2,
    /// left, right, center channels
    Channels3 = 3,
    /// front left, front right, back left, back right channels
    Channels4 = 4,
    /// front left, front right, front center,
    /// back/surround left, back/surround right channels
    Channels5 = 5,
    /// front left, front right, front center,
    /// LFE, back/surround left, back/surround right channels
    Channels6 = 6,
    /// front left, front right, front center,
    /// LFE, back center, side left, side right channels
    Channels7 = 7,
    /// front left, front right, front center,
    /// LFE, back left, back right, side left, side right channels
    Channels8 = 8,
}

impl From<Independent> for u8 {
    fn from(ch: Independent) -> Self {
        ch as u8
    }
}

impl From<Independent> for usize {
    fn from(ch: Independent) -> Self {
        ch as usize
    }
}

impl TryFrom<usize> for Independent {
    type Error = ();

    fn try_from(ch: usize) -> Result<Self, Self::Error> {
        match ch {
            1 => Ok(Self::Mono),
            2 => Ok(Self::Stereo),
            3 => Ok(Self::Channels3),
            4 => Ok(Self::Channels4),
            5 => Ok(Self::Channels5),
            6 => Ok(Self::Channels6),
            7 => Ok(Self::Channels7),
            8 => Ok(Self::Channels8),
            _ => Err(()),
        }
    }
}

/// How the channels are assigned in a FLAC frame
///
/// | Bits   | Channel Assingment |
/// |-------:|--------------------|
/// | `0000` | 1 mono channel
/// | `0001` | 2 independent channels
/// | `0010` | 3 independent channels
/// | `0011` | 4 independent channels
/// | `0100` | 5 independent channels
/// | `0101` | 6 independent channels
/// | `0110` | 7 independent channels
/// | `0111` | 8 independent channels
/// | `1000` | left channel, side channel
/// | `1001` | side channel, right channel
/// | `1010` | mid channel, side channel
/// | `1011` | invalid channel assignment
/// | `1100` | invalid channel assignment
/// | `1101` | invalid channel assignment
/// | `1110` | invalid channel assignment
/// | `1111` | invalid channel assignment
///
/// # Example
///
/// ```
/// use flac_codec::stream::{ChannelAssignment, Independent};
/// use bitstream_io::{BitReader, BitRead, BigEndian};
///
/// let data: &[u8] = &[
///     0b0000_100_0,             // channels + bps + pad
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// assert_eq!(
///     r.parse::<ChannelAssignment>().unwrap(),
///     ChannelAssignment::Independent(Independent::Mono),
/// );
/// ```
///
/// The samples in the side channel can be calculated like:
///
/// > sideᵢ = leftᵢ - rightᵢ
///
/// This requires that the side channel have one additional
/// bit-per-sample during decoding, since the difference
/// between the left and right channels could overflow
/// if the two are at opposite extremes.
///
/// And with a bit of math, we can see that:
///
/// > rightᵢ = leftᵢ - sideᵢ
///
/// or:
///
/// > leftᵢ = sideᵢ + rightᵢ
///
/// For transforming left-side and side-right assignments back
/// to left-right for output.
///
/// The samples in the mid channel can be calculated like:
///
/// > midᵢ = (leftᵢ + rightᵢ) ÷ 2
///
/// Mid-side assignment can be restored to left-right like:
///
/// > sumᵢ = midᵢ × 2 + |sideᵢ| % 2
/// >
/// > leftᵢ = (sumᵢ + sideᵢ) ÷ 2
/// >
/// > rightᵢ = (sumᵢ - sideᵢ) ÷ 2
///
/// The mid channel does *not* require any additional bits
/// to decode, since the average cannot exceed either channel.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ChannelAssignment {
    /// Channels are stored independently
    Independent(Independent),
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
            Self::Independent(c) => (*c).into(),
            _ => 2,
        }
    }
}

impl FromBitStream for ChannelAssignment {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read::<4, u8>()? {
            0b0000 => Ok(Self::Independent(Independent::Mono)),
            0b0001 => Ok(Self::Independent(Independent::Stereo)),
            0b0010 => Ok(Self::Independent(Independent::Channels3)),
            0b0011 => Ok(Self::Independent(Independent::Channels4)),
            0b0100 => Ok(Self::Independent(Independent::Channels5)),
            0b0101 => Ok(Self::Independent(Independent::Channels6)),
            0b0110 => Ok(Self::Independent(Independent::Channels7)),
            0b0111 => Ok(Self::Independent(Independent::Channels8)),
            0b1000 => Ok(Self::LeftSide),
            0b1001 => Ok(Self::SideRight),
            0b1010 => Ok(Self::MidSide),
            0b1011..=0b1111 => Err(Error::InvalidChannels),
            0b10000.. => unreachable!(), // 4-bit field
        }
    }
}

impl ToBitStream for ChannelAssignment {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        Ok(w.write::<4, u8>(match self {
            Self::Independent(Independent::Mono) => 0b0000,
            Self::Independent(Independent::Stereo) => 0b0001,
            Self::Independent(Independent::Channels3) => 0b0010,
            Self::Independent(Independent::Channels4) => 0b0011,
            Self::Independent(Independent::Channels5) => 0b0100,
            Self::Independent(Independent::Channels6) => 0b0101,
            Self::Independent(Independent::Channels7) => 0b0110,
            Self::Independent(Independent::Channels8) => 0b0111,
            Self::LeftSide => 0b1000,
            Self::SideRight => 0b1001,
            Self::MidSide => 0b1010,
        })?)
    }
}

/// The the possible bits-per-sample of a FLAC frame
///
/// Common bits-per-sample are stored as a 3-bit value,
/// while uncommon bits-per-sample are stored in the
/// STREAMINFO metadata block.
/// Bits-per-sample defined in the STREAMINFO metadata block
/// are only possible on a "non-subset" stream, which is
/// not streamable.
///
/// | Bits  | Bits-per-Sample |
/// |------:|-------------|
/// | `000` | get from STREAMINFO
/// | `001` | 8
/// | `010` | 12
/// | `011` | invalid
/// | `100` | 16
/// | `101` | 20
/// | `110` | 24
/// | `111` | 32
///
/// # Example
/// ```
/// use flac_codec::stream::BitsPerSample;
/// use bitstream_io::{BitReader, BitRead, BigEndian};
///
/// let data: &[u8] = &[
///     0b0000_100_0,             // channels + bps + pad
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// r.skip(4).unwrap();  // skip channel assignment
/// assert_eq!(
///     r.parse::<BitsPerSample>().unwrap(),
///     BitsPerSample::Bps16,
/// );
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BitsPerSample {
    /// Gets bps from STREAMINFO metadata block
    Streaminfo(SignedBitCount<32>),
    /// 8 bits-per-sample
    Bps8,
    /// 12 bits-per-sample
    Bps12,
    /// 16 bits-per-sample
    Bps16,
    /// 20 bits-per-sample
    Bps20,
    /// 24 bits-per-sample
    Bps24,
    /// 32 bits-per-sample
    Bps32,
}

impl BitsPerSample {
    const BPS8: SignedBitCount<32> = SignedBitCount::new::<8>();
    const BPS12: SignedBitCount<32> = SignedBitCount::new::<12>();
    const BPS16: SignedBitCount<32> = SignedBitCount::new::<16>();
    const BPS20: SignedBitCount<32> = SignedBitCount::new::<20>();
    const BPS24: SignedBitCount<32> = SignedBitCount::new::<24>();
    const BPS32: SignedBitCount<32> = SignedBitCount::new::<32>();

    /// Adds the given number of bits to this bit count, if possible.
    ///
    /// If the number of bits would overflow the maximum count,
    /// returns `None`.
    #[inline]
    pub fn checked_add(self, rhs: u32) -> Option<SignedBitCount<32>> {
        match self {
            Self::Streaminfo(c) => c.checked_add(rhs),
            Self::Bps8 => Self::BPS8.checked_add(rhs),
            Self::Bps12 => Self::BPS12.checked_add(rhs),
            Self::Bps16 => Self::BPS16.checked_add(rhs),
            Self::Bps20 => Self::BPS20.checked_add(rhs),
            Self::Bps24 => Self::BPS24.checked_add(rhs),
            Self::Bps32 => Self::BPS32.checked_add(rhs),
        }
    }
}

impl PartialEq<SignedBitCount<32>> for BitsPerSample {
    fn eq(&self, rhs: &SignedBitCount<32>) -> bool {
        match self {
            Self::Streaminfo(c) => c.eq(rhs),
            Self::Bps8 => Self::BPS8.eq(rhs),
            Self::Bps12 => Self::BPS12.eq(rhs),
            Self::Bps16 => Self::BPS16.eq(rhs),
            Self::Bps20 => Self::BPS20.eq(rhs),
            Self::Bps24 => Self::BPS24.eq(rhs),
            Self::Bps32 => Self::BPS32.eq(rhs),
        }
    }
}

impl From<BitsPerSample> for SignedBitCount<32> {
    #[inline]
    fn from(bps: BitsPerSample) -> Self {
        match bps {
            BitsPerSample::Streaminfo(c) => c,
            BitsPerSample::Bps8 => BitsPerSample::BPS8,
            BitsPerSample::Bps12 => BitsPerSample::BPS12,
            BitsPerSample::Bps16 => BitsPerSample::BPS16,
            BitsPerSample::Bps20 => BitsPerSample::BPS20,
            BitsPerSample::Bps24 => BitsPerSample::BPS24,
            BitsPerSample::Bps32 => BitsPerSample::BPS32,
        }
    }
}

impl From<BitsPerSample> for u32 {
    #[inline]
    fn from(bps: BitsPerSample) -> Self {
        match bps {
            BitsPerSample::Streaminfo(c) => c.into(),
            BitsPerSample::Bps8 => 8,
            BitsPerSample::Bps12 => 12,
            BitsPerSample::Bps16 => 16,
            BitsPerSample::Bps20 => 20,
            BitsPerSample::Bps24 => 24,
            BitsPerSample::Bps32 => 32,
        }
    }
}

impl From<SignedBitCount<32>> for BitsPerSample {
    #[inline]
    fn from(bps: SignedBitCount<32>) -> Self {
        match bps {
            Self::BPS8 => Self::Bps8,
            Self::BPS12 => Self::Bps12,
            Self::BPS16 => Self::Bps16,
            Self::BPS20 => Self::Bps20,
            Self::BPS24 => Self::Bps24,
            Self::BPS32 => Self::Bps32,
            bps => Self::Streaminfo(bps),
        }
    }
}

impl FromBitStreamUsing for BitsPerSample {
    type Context = Option<SignedBitCount<32>>;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        streaminfo_bps: Option<SignedBitCount<32>>,
    ) -> Result<Self, Error> {
        match r.read::<3, u8>()? {
            0b000 => Ok(Self::Streaminfo(
                streaminfo_bps.ok_or(Error::NonSubsetBitsPerSample)?,
            )),
            0b001 => Ok(Self::Bps8),
            0b010 => Ok(Self::Bps12),
            0b011 => Err(Error::InvalidBitsPerSample),
            0b100 => Ok(Self::Bps16),
            0b101 => Ok(Self::Bps20),
            0b110 => Ok(Self::Bps24),
            0b111 => Ok(Self::Bps32),
            0b1000.. => unreachable!(), // 3-bit field
        }
    }
}

impl FromBitStream for BitsPerSample {
    type Error = Error;

    #[inline]
    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        r.parse_using(None)
    }
}

impl ToBitStream for BitsPerSample {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<3, u8>(match self {
            Self::Streaminfo(_) => 0b000,
            Self::Bps8 => 0b001,
            Self::Bps12 => 0b010,
            Self::Bps16 => 0b100,
            Self::Bps20 => 0b101,
            Self::Bps24 => 0b110,
            Self::Bps32 => 0b111,
        })
    }
}

/// A frame number in the stream, as FLAC frames or samples
///
/// The frame number is stored as a UTF-8-like value
/// where the total number of bytes is encoded
/// in the initial byte.
///
/// | byte 0     | byte 1     | byte 2     | byte 3     | byte 4     | byte 5     | byte 6
/// |------------|------------|------------|------------|------------|------------|---------
/// | `0xxxxxxx` |            |            |            |            |            |
/// | `110xxxxx` | `10xxxxxx` |            |            |            |            |
/// | `1110xxxx` | `10xxxxxx` | `10xxxxxx` |            |            |            |
/// | `11110xxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` |            |            |
/// | `111110xx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` |            |
/// | `1111110x` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` |
/// | `11111110` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx`
///
/// The `x` bits are the frame number, encoded from most-significant
/// to least significant.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct FrameNumber(pub u64);

impl FrameNumber {
    /// Our maximum frame number
    const MAX_FRAME_NUMBER: u64 = (1 << 36) - 1;

    /// Attempt to increment frame number
    ///
    /// # Error
    ///
    /// Returns an error if the frame number is too large
    pub fn try_increment(&mut self) -> Result<(), Error> {
        if self.0 < Self::MAX_FRAME_NUMBER {
            self.0 += 1;
            Ok(())
        } else {
            Err(Error::ExcessiveFrameNumber)
        }
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
///
/// | Bits | Field   | Meaning
/// |-----:|---------|--------
/// | 1    | pad (0) |
/// | 6    | `type_` | subframe type and order (if any)
/// | 1    | has wasted bits | whether the subframe has wasted bits-per-sample
/// | (0+) | `wasted_bps`| the number of wasted bits-per-sample, as unary
///
/// "Wasted" bits is when all of the samples in a given subframe
/// have one or more `0` bits in the least-significant bits position.
/// In this case, the samples can all be safely right shifted
/// by that amount of wasted bits during encoding, and left
/// shifted by that amount of bits during decoding, without loss.
///
/// This is an uncommon case.
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

/// A subframe's type
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash, Ord, PartialOrd)]
pub enum SubframeType {
    /// A constant subframe
    Constant,
    /// A verbatim subframe
    Verbatim,
    /// A fixed subframe
    Fixed,
    /// An LPC subframe
    Lpc,
}

/// A subframe header's type and order
///
/// This is always a 6-bit field.
///
/// | Bits     | Type and Order
/// |----------|---------------
/// | `000000` | Constant
/// | `000001` | Verbatim
/// | `000010` to `000111` | reserved
/// | `001000` to `001100` | Fixed subframe with order `v - 8`
/// | `001101` to `011111` | reserved
/// | `100000` to `111111` | LPC subframe with order `v - 31`
#[derive(Debug, Eq, PartialEq)]
pub enum SubframeHeaderType {
    /// All samples are the same
    ///
    /// # Example
    /// ```
    /// use flac_codec::stream::SubframeHeaderType;
    /// use bitstream_io::{BitReader, BitRead, BigEndian};
    ///
    /// let data: &[u8] = &[0b0_000000_0];
    /// let mut r = BitReader::endian(data, BigEndian);
    /// r.skip(1).unwrap();  // pad bit
    /// assert_eq!(
    ///     r.parse::<SubframeHeaderType>().unwrap(),
    ///     SubframeHeaderType::Constant,
    /// );
    /// ```
    Constant,
    /// All samples as stored verbatim, without compression
    ///
    /// # Example
    /// ```
    /// use flac_codec::stream::SubframeHeaderType;
    /// use bitstream_io::{BitReader, BitRead, BigEndian};
    ///
    /// let data: &[u8] = &[0b0_000001_0];
    /// let mut r = BitReader::endian(data, BigEndian);
    /// r.skip(1).unwrap();  // pad bit
    /// assert_eq!(
    ///     r.parse::<SubframeHeaderType>().unwrap(),
    ///     SubframeHeaderType::Verbatim,
    /// );
    /// ```
    Verbatim,
    /// Samples are stored with one of a set of fixed LPC parameters
    ///
    /// # Example
    /// ```
    /// use flac_codec::stream::SubframeHeaderType;
    /// use bitstream_io::{BitReader, BitRead, BigEndian};
    ///
    /// let data: &[u8] = &[0b0_001100_0];
    /// let mut r = BitReader::endian(data, BigEndian);
    /// r.skip(1).unwrap();  // pad bit
    /// assert_eq!(
    ///     r.parse::<SubframeHeaderType>().unwrap(),
    ///     SubframeHeaderType::Fixed { order: 0b001100 - 8 },  // order = 4
    /// );
    /// ```
    Fixed {
        /// The predictor order, from 0..5
        order: u8,
    },
    /// Samples are stored with dynamic LPC parameters
    ///
    /// # Example
    /// ```
    /// use flac_codec::stream::SubframeHeaderType;
    /// use bitstream_io::{BitReader, BitRead, BigEndian};
    /// use std::num::NonZero;
    ///
    /// let data: &[u8] = &[0b0_100000_0];
    /// let mut r = BitReader::endian(data, BigEndian);
    /// r.skip(1).unwrap();  // pad bit
    /// assert_eq!(
    ///     r.parse::<SubframeHeaderType>().unwrap(),
    ///     SubframeHeaderType::Lpc {
    ///         order: NonZero::new(0b100000 - 31).unwrap(),  // order = 1
    ///     },
    /// );
    /// ```
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
#[derive(Debug, Eq, PartialEq)]
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
///
/// A FLAC frame consists of a header, one or more subframes
/// (each corresponding to a different channel), and a CRC-16
/// checksum.
///
/// # Example
///
/// ```
/// use flac_codec::stream::{
///     Frame, FrameHeader, BlockSize, SampleRate, ChannelAssignment,
///     Independent, BitsPerSample, FrameNumber, SubframeWidth, Subframe,
/// };
///
/// let mut data: &[u8] = &[
///     // frame header
///     0xff, 0xf8, 0x69, 0x08, 0x00, 0x13, 0x64,
///     // subframe
///     0x00, 0x00, 0x00,
///     // CRC-16
///     0xd3, 0x3b,
/// ];
///
/// assert_eq!(
///     Frame::read_subset(&mut data).unwrap(),
///     Frame {
///         header: FrameHeader {
///             blocking_strategy: false,
///             block_size: BlockSize::Uncommon8(20),
///             sample_rate: SampleRate::Hz44100,
///             channel_assignment: ChannelAssignment::Independent(
///                 Independent::Mono
///             ),
///             bits_per_sample: BitsPerSample::Bps16,
///             frame_number: FrameNumber(0),
///         },
///         subframes: vec![
///             SubframeWidth::Common(
///                 Subframe::Constant {
///                     block_size: 20,
///                     sample: 0x00_00,
///                     wasted_bps: 0,
///                 },
///             )
///         ],
///     },
/// );
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct Frame {
    /// The FLAC frame's header
    pub header: FrameHeader,
    /// A FLAC frame's sub-frames
    pub subframes: Vec<SubframeWidth>,
}

impl Frame {
    fn read_inner<R, F>(reader: &mut R, read_header: F) -> Result<Self, Error>
    where
        R: std::io::Read,
        F: FnOnce(&mut CrcReader<&mut R, Crc16>) -> Result<FrameHeader, Error>,
    {
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc16_reader: CrcReader<_, Crc16> = CrcReader::new(reader.by_ref());

        let header = read_header(crc16_reader.by_ref())?;

        let mut reader = BitReader::endian(crc16_reader.by_ref(), BigEndian);

        let subframes = match header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => (0..total_channels as u8)
                .map(|_| {
                    reader
                        .parse_using::<Subframe<i32>>((
                            header.block_size.into(),
                            header.bits_per_sample.into(),
                        ))
                        .map(SubframeWidth::Common)
                })
                .collect::<Result<Vec<_>, _>>()?,
            ChannelAssignment::LeftSide => vec![
                reader
                    .parse_using((header.block_size.into(), header.bits_per_sample.into()))
                    .map(SubframeWidth::Common)?,
                match header.bits_per_sample.checked_add(1) {
                    Some(side_bps) => reader
                        .parse_using((header.block_size.into(), side_bps))
                        .map(SubframeWidth::Common)?,
                    None => reader
                        .parse_using((
                            header.block_size.into(),
                            SignedBitCount::from(header.bits_per_sample)
                                .checked_add(1)
                                .unwrap(),
                        ))
                        .map(SubframeWidth::Wide)?,
                },
            ],
            ChannelAssignment::SideRight => vec![
                match header.bits_per_sample.checked_add(1) {
                    Some(side_bps) => reader
                        .parse_using((header.block_size.into(), side_bps))
                        .map(SubframeWidth::Common)?,
                    None => reader
                        .parse_using((
                            header.block_size.into(),
                            SignedBitCount::from(header.bits_per_sample)
                                .checked_add(1)
                                .unwrap(),
                        ))
                        .map(SubframeWidth::Wide)?,
                },
                reader
                    .parse_using((header.block_size.into(), header.bits_per_sample.into()))
                    .map(SubframeWidth::Common)?,
            ],
            ChannelAssignment::MidSide => vec![
                reader
                    .parse_using((header.block_size.into(), header.bits_per_sample.into()))
                    .map(SubframeWidth::Common)?,
                match header.bits_per_sample.checked_add(1) {
                    Some(side_bps) => reader
                        .parse_using((header.block_size.into(), side_bps))
                        .map(SubframeWidth::Common)?,
                    None => reader
                        .parse_using((
                            header.block_size.into(),
                            SignedBitCount::from(header.bits_per_sample)
                                .checked_add(1)
                                .unwrap(),
                        ))
                        .map(SubframeWidth::Wide)?,
                },
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

    /// Reads new frame from the given reader
    #[inline]
    pub fn read<R: std::io::Read>(reader: &mut R, streaminfo: &Streaminfo) -> Result<Self, Error> {
        Self::read_inner(reader, |r| FrameHeader::read(r, streaminfo))
    }

    /// Reads new frame from the given reader
    ///
    /// Subset files are streamable FLAC files whose decoding
    /// parameters are fully contained within each frame header.
    #[inline]
    pub fn read_subset<R: std::io::Read>(reader: &mut R) -> Result<Self, Error> {
        Self::read_inner(reader, |r| FrameHeader::read_subset(r))
    }

    fn write_inner<W, F>(&self, write_header: F, writer: &mut W) -> Result<(), Error>
    where
        W: std::io::Write,
        F: FnOnce(&mut CrcWriter<&mut W, Crc16>, &FrameHeader) -> Result<(), Error>,
    {
        use bitstream_io::{BigEndian, BitWriter};
        use std::io::Write;

        let mut crc16_writer: CrcWriter<_, Crc16> = CrcWriter::new(writer.by_ref());

        write_header(crc16_writer.by_ref(), &self.header)?;

        let mut writer = BitWriter::endian(crc16_writer.by_ref(), BigEndian);

        match self.header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => {
                assert_eq!(total_channels as usize, self.subframes.len());

                for subframe in &self.subframes {
                    // independent subframes should always be standard width
                    if let SubframeWidth::Common(subframe) = subframe {
                        writer.build_using(subframe, self.header.bits_per_sample.into())?;
                    }
                }
            }
            ChannelAssignment::LeftSide => match self.subframes.as_slice() {
                [left, side] => {
                    if let SubframeWidth::Common(left) = left {
                        writer.build_using(left, self.header.bits_per_sample.into())?;
                    }

                    match side {
                        SubframeWidth::Common(side) => {
                            writer.build_using(
                                side,
                                self.header
                                    .bits_per_sample
                                    .checked_add(1)
                                    .ok_or(Error::ExcessiveBps)?,
                            )?;
                        }
                        SubframeWidth::Wide(side) => {
                            writer.build_using(
                                side,
                                SignedBitCount::from(self.header.bits_per_sample)
                                    .checked_add(1)
                                    .ok_or(Error::ExcessiveBps)?,
                            )?;
                        }
                    }
                }
                _ => panic!("incorrect subframe count for left-side"),
            },
            ChannelAssignment::SideRight => match self.subframes.as_slice() {
                [side, right] => {
                    match side {
                        SubframeWidth::Common(side) => {
                            writer.build_using(
                                side,
                                self.header
                                    .bits_per_sample
                                    .checked_add(1)
                                    .ok_or(Error::ExcessiveBps)?,
                            )?;
                        }
                        SubframeWidth::Wide(side) => {
                            writer.build_using(
                                side,
                                SignedBitCount::from(self.header.bits_per_sample)
                                    .checked_add(1)
                                    .ok_or(Error::ExcessiveBps)?,
                            )?;
                        }
                    }

                    if let SubframeWidth::Common(right) = right {
                        writer.build_using(right, self.header.bits_per_sample.into())?;
                    }
                }
                _ => panic!("incorrect subframe count for left-side"),
            },
            ChannelAssignment::MidSide => match self.subframes.as_slice() {
                [mid, side] => {
                    if let SubframeWidth::Common(mid) = mid {
                        writer.build_using(mid, self.header.bits_per_sample.into())?;
                    }

                    match side {
                        SubframeWidth::Common(side) => {
                            writer.build_using(
                                side,
                                self.header
                                    .bits_per_sample
                                    .checked_add(1)
                                    .ok_or(Error::ExcessiveBps)?,
                            )?;
                        }
                        SubframeWidth::Wide(side) => {
                            writer.build_using(
                                side,
                                SignedBitCount::from(self.header.bits_per_sample)
                                    .checked_add(1)
                                    .ok_or(Error::ExcessiveBps)?,
                            )?;
                        }
                    }
                }
                _ => panic!("incorrect subframe count for left-side"),
            },
        }

        let crc16: u16 = writer.aligned_writer()?.checksum().into();
        writer.write_from(crc16)?;

        Ok(())
    }

    /// Writes frame to the given writer
    pub fn write<W: std::io::Write>(
        &self,
        streaminfo: &Streaminfo,
        writer: &mut W,
    ) -> Result<(), Error> {
        self.write_inner(|w, header| header.write(w, streaminfo), writer)
    }

    /// Writes frame to the given writer
    ///
    /// Subset files are streamable FLAC files whose encoding
    /// parameters are fully contained within each frame header.
    #[inline]
    pub fn write_subset<W: std::io::Write>(&self, writer: &mut W) -> Result<(), Error> {
        self.write_inner(|w, header| header.write_subset(w), writer)
    }
}

/// A 32 or 64-bit FLAC file subframe
#[derive(Debug, Eq, PartialEq)]
pub enum SubframeWidth {
    /// A common 32-bit subframe
    Common(Subframe<i32>),
    /// A rare 64-bit subframe
    Wide(Subframe<i64>),
}

/// A FLAC's frame's subframe, one per channel
#[derive(Debug, Eq, PartialEq)]
pub enum Subframe<I> {
    /// A CONSTANT subframe, in which all samples are identical
    ///
    /// This is typically for long stretches of silence,
    /// or for when both channels are identical in a stereo stream.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::stream::Subframe;
    /// use bitstream_io::{BitReader, BitRead, BigEndian, SignedBitCount};
    ///
    /// let data: &[u8] = &[
    ///     0x00,        // subframe header
    ///     0x00, 0x00,  // subframe data
    /// ];
    ///
    /// let mut r = BitReader::endian(data, BigEndian);
    ///
    /// assert_eq!(
    ///     r.parse_using::<Subframe<i32>>((20, SignedBitCount::new::<16>())).unwrap(),
    ///     Subframe::Constant {
    ///         // taken from context
    ///         block_size: 20,
    ///         // constant subframes always have exactly one sample
    ///         // this sample's size is a signed 16-bit value
    ///         // taken from the subframe signed bit count
    ///         sample: 0x00_00,
    ///         // wasted bits-per-sample is taken from the subframe header
    ///         wasted_bps: 0,
    ///     },
    /// );
    /// ```
    Constant {
        /// the subframe's block size in samples
        block_size: u16,
        /// The subframe's sample
        sample: I,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
    /// A VERBATIM subframe, in which all samples are stored uncompressed
    ///
    /// This is for random noise which does not compress well
    /// by any other method.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::stream::Subframe;
    /// use bitstream_io::{BitReader, BitRead, BigEndian, SignedBitCount};
    ///
    /// let data: &[u8] = &[
    ///     0x02,  // subframe header
    ///     // subframe data
    ///     0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04,
    ///     0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x08, 0x00, 0x09,
    ///     0x00, 0x0a, 0x00, 0x0b, 0x00, 0x0c, 0x00, 0x0d, 0x00, 0x0e,
    ///     0x00, 0x0f, 0x00, 0x10, 0x00, 0x11, 0x00, 0x12, 0x00, 0x13,
    /// ];
    ///
    /// let mut r = BitReader::endian(data, BigEndian);
    ///
    /// assert_eq!(
    ///     r.parse_using::<Subframe<i32>>((20, SignedBitCount::new::<16>())).unwrap(),
    ///     Subframe::Verbatim {
    ///         // the total number of samples equals the block size
    ///         // (20 in this case)
    ///         // each sample is a signed 16-bit value,
    ///         // taken from the subframe signed bit count
    ///         samples: vec![
    ///             0x00, 0x01, 0x02, 0x03, 0x04,
    ///             0x05, 0x06, 0x07, 0x08, 0x09,
    ///             0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
    ///             0x0f, 0x10, 0x11, 0x12, 0x13,
    ///         ],
    ///         // wasted bits-per-sample is taken from the subframe header
    ///         wasted_bps: 0,
    ///     },
    /// );
    /// ```
    Verbatim {
        /// The subframe's samples
        samples: Vec<I>,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
    /// A FIXED subframe, encoded with a fixed set of parameters
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::stream::{Subframe, Residuals, ResidualPartition};
    /// use bitstream_io::{BitReader, BitRead, BigEndian, BitCount, SignedBitCount};
    ///
    /// let data: &[u8] = &[
    ///     0x18,  // subframe header
    ///     // warm-up samples
    ///     0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03,
    ///     // residuals
    ///     0x00, 0x3f, 0xff, 0xc0,
    /// ];
    ///
    /// let mut r = BitReader::endian(data, BigEndian);
    ///
    /// assert_eq!(
    ///     r.parse_using::<Subframe<i32>>((20, SignedBitCount::new::<16>())).unwrap(),
    ///     Subframe::Fixed {
    ///         // predictor order is determined from subframe header
    ///         order: 4,
    ///         // the total number of warm-up samples equals the predictor order (4)
    ///         // each warm-up sample is a signed 16-bit value,
    ///         // taken from the subframe signed bit count
    ///         warm_up: vec![0x00, 0x01, 0x02, 0x03],
    ///         // the total number of residuals equals the block size
    ///         // minus the predictor order,
    ///         // which is 20 - 4 = 16 in this case
    ///         residuals: Residuals::Method0 {
    ///             partitions: vec![
    ///                 ResidualPartition::Standard {
    ///                     rice: BitCount::new::<0>(),
    ///                     residuals: vec![0; 16],
    ///                 }
    ///             ],
    ///         },
    ///         // wasted bits-per-sample is taken from the subframe header
    ///         wasted_bps: 0,
    ///     },
    /// );
    /// ```
    Fixed {
        /// The subframe's predictor order from 0 to 4 (inclusive)
        order: u8,
        /// The subframe's warm-up samples (one per order)
        warm_up: Vec<I>,
        /// The subframe's residuals
        residuals: Residuals<I>,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
    /// An LPC subframe, encoded with a variable set of parameters
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::stream::{Subframe, Residuals, ResidualPartition};
    /// use bitstream_io::{BitReader, BitRead, BigEndian, BitCount, SignedBitCount};
    /// use std::num::NonZero;
    ///
    /// let data: &[u8] = &[
    ///     0x40,  // subframe header
    ///     // warm-up sample
    ///     0x00, 0x00,
    ///     // precision + shift + coefficient
    ///     0b1011_0101, 0b1_0111110, 0b00101_000,
    ///     // residuals
    ///     0x02, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x80,
    /// ];
    ///
    /// let mut r = BitReader::endian(data, BigEndian);
    ///
    /// assert_eq!(
    ///     r.parse_using::<Subframe<i32>>((20, SignedBitCount::new::<16>())).unwrap(),
    ///     Subframe::Lpc {
    ///         // predictor order is determined from the subframe header
    ///         order: NonZero::new(1).unwrap(),
    ///         // the total number of warm-up samples equals the predictor order (1)
    ///         // each warm-up sample is a signed 16-bit value,
    ///         // taken from the subframe signed bit count
    ///         warm_up: vec![0x00],
    ///         // precision is a 4 bit value, plus one
    ///         precision: SignedBitCount::new::<{0b1011 + 1}>(),  // 12 bits
    ///         // shift is a 5 bit value
    ///         shift: 0b0101_1,  // 11
    ///         // the total number of coefficients equals the predictor order (1)
    ///         // size of each coefficient is a signed 12-bit value, from precision
    ///         coefficients: vec![0b0111110_00101],  // 1989
    ///         // the total number of residuals equals the block size
    ///         // minus the predictor order,
    ///         // which is 20 - 1 = 19 in this case
    ///         residuals: Residuals::Method0 {
    ///             partitions: vec![
    ///                 ResidualPartition::Standard {
    ///                     rice: BitCount::new::<1>(),
    ///                     residuals: vec![
    ///                          1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ///                          2, 2, 2, 2, 2, 2, 2, 2, 2
    ///                     ],
    ///                 }
    ///             ],
    ///         },
    ///         // wasted bits-per-sample is taken from the subframe header
    ///         wasted_bps: 0,
    ///     },
    /// );
    /// ```
    Lpc {
        /// The subframe's predictor order
        order: NonZero<u8>,
        /// The subframe's warm-up samples (one per order)
        warm_up: Vec<I>,
        /// The subframe's QLP precision
        precision: SignedBitCount<15>,
        /// The subframe's QLP shift
        shift: u32,
        /// The subframe's QLP coefficients (one per order)
        coefficients: Vec<i32>,
        /// The subframe's residuals
        residuals: Residuals<I>,
        /// Any wasted bits-per-sample
        wasted_bps: u32,
    },
}

impl<I> Subframe<I> {
    /// Our subframe type
    pub fn subframe_type(&self) -> SubframeType {
        match self {
            Self::Constant { .. } => SubframeType::Constant,
            Self::Verbatim { .. } => SubframeType::Verbatim,
            Self::Fixed { .. } => SubframeType::Fixed,
            Self::Lpc { .. } => SubframeType::Lpc,
        }
    }
}

impl<I: SignedInteger> Subframe<I> {
    /// Decodes subframe to samples
    ///
    /// Note that decoding subframes to samples using this method
    /// is intended for analysis purposes.  The [`crate::decode`]
    /// module's decoders are preferred for general-purpose
    /// decoding as they perform fewer temporary allocations.
    pub fn decode(&self) -> Box<dyn Iterator<Item = I> + '_> {
        fn predict<I: SignedInteger>(coefficients: &[i64], qlp_shift: u32, channel: &mut [I]) {
            for split in coefficients.len()..channel.len() {
                let (predicted, residuals) = channel.split_at_mut(split);

                residuals[0] += I::from_i64(
                    predicted
                        .iter()
                        .rev()
                        .zip(coefficients)
                        .map(|(x, y)| (*x).into() * y)
                        .sum::<i64>()
                        >> qlp_shift,
                );
            }
        }

        match self {
            Self::Constant {
                sample,
                block_size,
                wasted_bps,
            } => Box::new((0..*block_size).map(move |_| *sample << *wasted_bps)),
            Self::Verbatim {
                samples,
                wasted_bps,
            } => Box::new(samples.iter().map(move |sample| *sample << *wasted_bps)),
            Self::Fixed {
                order,
                warm_up,
                residuals,
                wasted_bps,
            } => {
                let mut samples = warm_up.clone();
                samples.extend(residuals.residuals());
                predict(
                    SubframeHeaderType::FIXED_COEFFS[*order as usize],
                    0,
                    &mut samples,
                );
                Box::new(samples.into_iter().map(move |sample| sample << *wasted_bps))
            }
            Self::Lpc {
                warm_up,
                coefficients,
                residuals,
                wasted_bps,
                shift,
                ..
            } => {
                let mut samples = warm_up.clone();
                samples.extend(residuals.residuals());
                predict(
                    &coefficients
                        .iter()
                        .copied()
                        .map(i64::from)
                        .collect::<Vec<_>>(),
                    *shift,
                    &mut samples,
                );
                Box::new(samples.into_iter().map(move |sample| sample << *wasted_bps))
            }
        }
    }
}

fn read_subframe<const MAX: u32, R, I>(
    r: &mut R,
    block_size: u16,
    bits_per_sample: SignedBitCount<MAX>,
) -> Result<Subframe<I>, Error>
where
    R: BitRead + ?Sized,
    I: SignedInteger,
{
    match r.parse()? {
        SubframeHeader {
            type_: SubframeHeaderType::Constant,
            wasted_bps,
        } => Ok(Subframe::Constant {
            block_size,
            sample: r.read_signed_counted(
                bits_per_sample
                    .checked_sub::<MAX>(wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?,
            )?,
            wasted_bps,
        }),
        SubframeHeader {
            type_: SubframeHeaderType::Verbatim,
            wasted_bps,
        } => {
            let effective_bps = bits_per_sample
                .checked_sub::<MAX>(wasted_bps)
                .ok_or(Error::ExcessiveWastedBits)?;

            Ok(Subframe::Verbatim {
                samples: (0..block_size)
                    .map(|_| r.read_signed_counted::<MAX, I>(effective_bps))
                    .collect::<Result<Vec<_>, _>>()?,
                wasted_bps,
            })
        }
        SubframeHeader {
            type_: SubframeHeaderType::Fixed { order },
            wasted_bps,
        } => {
            let effective_bps = bits_per_sample
                .checked_sub::<MAX>(wasted_bps)
                .ok_or(Error::ExcessiveWastedBits)?;

            Ok(Subframe::Fixed {
                order,
                warm_up: (0..order)
                    .map(|_| r.read_signed_counted::<MAX, I>(effective_bps))
                    .collect::<Result<Vec<_>, _>>()?,
                residuals: r.parse_using((block_size.into(), order.into()))?,
                wasted_bps,
            })
        }
        SubframeHeader {
            type_: SubframeHeaderType::Lpc { order },
            wasted_bps,
        } => {
            let effective_bps = bits_per_sample
                .checked_sub::<MAX>(wasted_bps)
                .ok_or(Error::ExcessiveWastedBits)?;

            let warm_up = (0..order.get())
                .map(|_| r.read_signed_counted::<MAX, I>(effective_bps))
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

            Ok(Subframe::Lpc {
                order,
                warm_up,
                precision,
                shift,
                coefficients,
                residuals: r.parse_using((block_size.into(), order.get().into()))?,
                wasted_bps,
            })
        }
    }
}

impl FromBitStreamUsing for Subframe<i32> {
    type Context = (u16, SignedBitCount<32>);
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        (block_size, bits_per_sample): (u16, SignedBitCount<32>),
    ) -> Result<Self, Error> {
        read_subframe(r, block_size, bits_per_sample)
    }
}

impl FromBitStreamUsing for Subframe<i64> {
    type Context = (u16, SignedBitCount<33>);
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        (block_size, bits_per_sample): (u16, SignedBitCount<33>),
    ) -> Result<Self, Error> {
        read_subframe(r, block_size, bits_per_sample)
    }
}

fn write_subframe<const MAX: u32, W, I>(
    w: &mut W,
    bits_per_sample: SignedBitCount<MAX>,
    subframe: &Subframe<I>,
) -> Result<(), Error>
where
    W: BitWrite + ?Sized,
    I: SignedInteger,
{
    match subframe {
        Subframe::Constant {
            sample, wasted_bps, ..
        } => {
            w.build(&SubframeHeader {
                type_: SubframeHeaderType::Constant,
                wasted_bps: *wasted_bps,
            })?;

            w.write_signed_counted(
                bits_per_sample
                    .checked_sub::<MAX>(*wasted_bps)
                    .ok_or(Error::ExcessiveWastedBits)?,
                *sample,
            )?;

            Ok(())
        }
        Subframe::Verbatim {
            samples,
            wasted_bps,
        } => {
            let effective_bps = bits_per_sample
                .checked_sub::<MAX>(*wasted_bps)
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
        Subframe::Fixed {
            order,
            warm_up,
            residuals,
            wasted_bps,
        } => {
            assert_eq!(*order as usize, warm_up.len());

            let effective_bps = bits_per_sample
                .checked_sub::<MAX>(*wasted_bps)
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
        Subframe::Lpc {
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
                .checked_sub::<MAX>(*wasted_bps)
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

impl ToBitStreamUsing for Subframe<i32> {
    type Context = SignedBitCount<32>;
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        bits_per_sample: SignedBitCount<32>,
    ) -> Result<(), Error> {
        write_subframe(w, bits_per_sample, self)
    }
}

impl ToBitStreamUsing for Subframe<i64> {
    type Context = SignedBitCount<33>;
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        bits_per_sample: SignedBitCount<33>,
    ) -> Result<(), Error> {
        write_subframe(w, bits_per_sample, self)
    }
}

/// Residual values for FIXED or LPC subframes
///
/// | Bits | Meaning |
/// |-----:|---------|
/// | 2    | residual coding method
/// | 4    | partition order
/// |      | residual partition₀
/// |      | (residual partition₁)
/// |      | ⋮
///
/// The residual coding method can be 0 or 1.
/// A coding method of 0 means 4-bit Rice parameters
/// in residual partitions.  A coding method of 5
/// means 5-bit Rice parameters in residual partitions
/// (method 0 is the common case).
///
/// The number of residual partitions equals
/// 2ⁿ where n is the partion order.
///
/// # Example
/// ```
/// use flac_codec::stream::{Residuals, ResidualPartition};
/// use bitstream_io::{BitReader, BitRead, BigEndian, BitCount};
///
/// let data: &[u8] = &[
///     0b00_0000_00,  // coding method + partition order + partition
///     // residual partition
///     0b01010001,
///     0b00010001,
///     0b00010001,
///     0b00010001,
///     0b00010001,
///     0b00010001,
///     0b00010001,
///     0b00010001,
///     0b00010001,
///     0b00010000,
///     0b00000000,
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
///
/// assert_eq!(
///     r.parse_using::<Residuals<i32>>((20, 1)).unwrap(),
///     // coding method = 0b00
///     // partition order = 0b0000, or 1 partition
///     Residuals::Method0 {
///         partitions: vec![
///             ResidualPartition::Standard {
///                 rice: BitCount::new::<1>(),
///                 residuals: vec![
///                      1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
///                      2, 2, 2, 2, 2, 2, 2, 2, 2
///                 ],
///             }
///         ],
///     },
/// );
/// ```
#[derive(Debug, Eq, PartialEq)]
pub enum Residuals<I> {
    /// Coding method 0
    Method0 {
        /// The residual partitions
        partitions: Vec<ResidualPartition<0b1111, I>>,
    },
    /// Coding method 1
    Method1 {
        /// The residual partitions
        partitions: Vec<ResidualPartition<0b11111, I>>,
    },
}

impl<I: SignedInteger> Residuals<I> {
    /// Iterates over all individual residual values
    fn residuals(&self) -> Box<dyn Iterator<Item = I> + '_> {
        match self {
            Self::Method0 { partitions } => Box::new(partitions.iter().flat_map(|p| p.residuals())),
            Self::Method1 { partitions } => Box::new(partitions.iter().flat_map(|p| p.residuals())),
        }
    }
}

impl<I: SignedInteger> FromBitStreamUsing for Residuals<I> {
    type Context = (usize, usize);
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        (block_size, predictor_order): (usize, usize),
    ) -> Result<Self, Error> {
        fn read_partitions<const RICE_MAX: u32, R: BitRead + ?Sized, I: SignedInteger>(
            reader: &mut R,
            (block_size, predictor_order): (usize, usize),
        ) -> Result<Vec<ResidualPartition<RICE_MAX, I>>, Error> {
            let partition_order = reader.read::<4, u32>()?;
            let partition_count = 1 << partition_order;

            (0..partition_count)
                .map(|p| {
                    reader.parse_using(
                        (block_size / partition_count)
                            .checked_sub(if p == 0 { predictor_order } else { 0 })
                            .ok_or(Error::InvalidPartitionOrder)?,
                    )
                })
                .collect()
        }

        match r.read::<2, u8>()? {
            0 => Ok(Self::Method0 {
                partitions: read_partitions::<0b1111, R, I>(r, (block_size, predictor_order))?,
            }),
            1 => Ok(Self::Method1 {
                partitions: read_partitions::<0b11111, R, I>(r, (block_size, predictor_order))?,
            }),
            _ => Err(Error::InvalidCodingMethod),
        }
    }
}

impl<I: SignedInteger> ToBitStream for Residuals<I> {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        fn write_partitions<const RICE_MAX: u32, W: BitWrite + ?Sized, I: SignedInteger>(
            writer: &mut W,
            partitions: &[ResidualPartition<RICE_MAX, I>],
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
///
/// Each partition consists of a Rice parameter
/// followed by an optional escape code and signed residual values.
/// The number of bits to read for the Rice parameters
/// depends on if we're using coding method 0 or 1.
/// If the Rice parameter equals the maximum,
/// it means the partition is escaped in some way
/// (this is an uncommon case).
///
/// | Bits    | Meaning |
/// |--------:|---------|
/// | 4 or 5  | Rice parameter
/// | (5)     | escape code if parameter is `1111` or `11111`
///
/// The total number of residuals in the partition is:
///
/// > block size ÷ partition count - predictor order
///
/// for the first partition, and:
///
/// > block size ÷ partition count
///
/// for subsequent partitions.
///
/// If the partition is escaped, we read an additional 5 bit value
/// to determine the size of each signed residual in the partition.
/// If the *escape code* is 0, all the residuals in the partition are 0
/// (this is an even more uncommon case).
///
/// # Example
/// ```
/// use flac_codec::stream::ResidualPartition;
/// use bitstream_io::{BitReader, BitRead, BigEndian, BitCount};
///
/// let data: &[u8] = &[
///     0b0001_01_0_0,  // Rice code + residuals
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
///     0b01_0_001_0_0,
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// assert_eq!(
///     r.parse_using::<ResidualPartition<0b1111, i32>>(19).unwrap(),
///     ResidualPartition::Standard {
///         rice: BitCount::new::<0b0001>(),
///         residuals: vec![
///              1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
///              2, 2, 2, 2, 2, 2, 2, 2, 2
///         ],
///     },
/// );
/// ```
///
/// Each individual residual is a unary value with a stop bit of 1
/// for the most-significant bits, followed by "Rice" number of
/// bits as the least significant bits, combined into a single
/// unsigned value.
///
/// Unary-encoding is simply counting the number of 0 bits
/// before the next 1 bit:
///
/// | Bits    | Value |
/// |--------:|-------|
/// | `1`     | 0
/// | `01`    | 1
/// | `001`   | 2
/// | `0001`  | 3
/// | `00001` | 4
/// | ⋮       |
///
/// Unlike regular twos-complement signed values, individual residuals
/// are stored with the sign in the *least* significant bit position.
/// They can be transformed from unsigned to signed like:
/// ```
/// fn unsigned_to_signed(unsigned: u32) -> i32 {
///     if (unsigned & 1) == 1 {
///         // negative residual
///         -((unsigned >> 1) as i32) - 1
///     } else {
///         // positive residual
///         (unsigned >> 1) as i32
///     }
/// }
/// ```
///
/// In our example, above, the Rice parameter happens to be 1
/// and all the sign bits happen to be 0, so the value of each
/// signed residual is simply its preceding unary value
/// (which are all `01` or `001`, meaning 1 and 2).
///
/// As one can see, the smaller the value each residual has,
/// the smaller it can be when written to disk.
/// And the key to making residual values small is to choose
/// predictor coefficients which best match the input signal.
/// The more accurate the prediction, the less difference
/// there is between the predicted values and the actual
/// values - which means smaller residuals - which means
/// better compression.
#[derive(Debug, Eq, PartialEq)]
pub enum ResidualPartition<const RICE_MAX: u32, I> {
    /// A standard residual partition
    Standard {
        /// The partition's Rice parameter
        rice: BitCount<RICE_MAX>,
        /// The partition's residuals
        residuals: Vec<I>,
    },
    /// An escaped residual partition
    Escaped {
        /// The size of each residual in bits
        escape_size: SignedBitCount<0b11111>,
        /// The partition's residuals
        residuals: Vec<I>,
    },
    /// A partition in which all residuals are 0
    Constant {
        /// The length of the partition in samples
        partition_len: usize,
    },
}

impl<const RICE_MAX: u32, I: SignedInteger> ResidualPartition<RICE_MAX, I> {
    fn residuals(&self) -> Box<dyn Iterator<Item = I> + '_> {
        match self {
            Self::Standard { residuals, .. } | Self::Escaped { residuals, .. } => {
                Box::new(residuals.iter().copied())
            }
            Self::Constant { partition_len } => {
                Box::new(std::iter::repeat_n(I::default(), *partition_len))
            }
        }
    }
}

impl<const RICE_MAX: u32, I: SignedInteger> FromBitStreamUsing for ResidualPartition<RICE_MAX, I> {
    type Context = usize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, partition_len: usize) -> Result<Self, Error> {
        match r.parse::<ResidualPartitionHeader<RICE_MAX>>()? {
            ResidualPartitionHeader::Standard { rice } => Ok(Self::Standard {
                residuals: (0..partition_len)
                    .map(|_| {
                        let msb = r.read_unary::<1>()?;
                        let lsb = r.read_counted::<RICE_MAX, u32>(rice)?;
                        let unsigned = (msb << u32::from(rice)) | lsb;
                        Ok::<_, Error>(if (unsigned & 1) == 1 {
                            -(I::from_u32(unsigned >> 1)) - I::ONE
                        } else {
                            I::from_u32(unsigned >> 1)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                rice,
            }),
            ResidualPartitionHeader::Escaped { escape_size } => Ok(Self::Escaped {
                residuals: (0..partition_len)
                    .map(|_| r.read_signed_counted(escape_size))
                    .collect::<Result<Vec<_>, _>>()?,
                escape_size,
            }),
            ResidualPartitionHeader::Constant => Ok(Self::Constant { partition_len }),
        }
    }
}

impl<const RICE_MAX: u32, I: SignedInteger> ToBitStream for ResidualPartition<RICE_MAX, I> {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        match self {
            Self::Standard { residuals, rice } => {
                w.build(&ResidualPartitionHeader::Standard { rice: *rice })?;

                let mask = rice.mask_lsb();

                for residual in residuals {
                    let (msb, lsb) = mask(if residual.is_negative() {
                        (((-*residual).to_u32() - 1) << 1) + 1
                    } else {
                        (*residual).to_u32() << 1
                    });
                    w.write_unary::<1>(msb)?;
                    w.write_checked(lsb)?;
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
            Self::Constant { .. } => Ok(w.build(&ResidualPartitionHeader::<RICE_MAX>::Constant)?),
        }
    }
}
