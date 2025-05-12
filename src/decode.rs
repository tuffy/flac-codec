// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For decoding FLAC files to PCM samples

use crate::Error;
use crate::audio::Frame;
use crate::metadata::{SeekTable, Streaminfo};
use bitstream_io::{BitRead, SignedBitCount};
use std::collections::VecDeque;
use std::num::NonZero;

/// A FLAC reader
pub struct Reader<R, E> {
    decoder: Decoder<R>,
    endianness: std::marker::PhantomData<E>,
    buf: VecDeque<u8>,
}

impl<R: std::io::Read, E> Reader<R, E> {
    /// Opens new FLAC reader
    pub fn new(reader: R) -> Result<Self, Error> {
        Ok(Self {
            decoder: Decoder::new(reader)?,
            endianness: std::marker::PhantomData,
            buf: VecDeque::default(),
        })
    }

    /// Returns channel count
    ///
    /// From 1 to 8
    pub fn channel_count(&self) -> NonZero<u8> {
        self.decoder.streaminfo.channels
    }

    /// Returns sample rate, in Hz
    pub fn sample_rate(&self) -> u32 {
        self.decoder.streaminfo.sample_rate
    }

    /// Returns decoder's bits-per-sample
    ///
    /// From 1 to 32
    pub fn bits_per_sample(&self) -> SignedBitCount<32> {
        self.decoder.streaminfo.bits_per_sample
    }

    /// Returns total number of channel-independent samples, if known
    pub fn total_samples(&self) -> Option<NonZero<u64>> {
        self.decoder.streaminfo.total_samples
    }

    /// Returns MD5 of entire stream, if known
    pub fn md5(&self) -> Option<&[u8; 16]> {
        self.decoder.streaminfo.md5.as_ref()
    }
}

impl<R: std::io::Read, E: crate::audio::Endianness> std::io::Read for Reader<R, E> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.buf.is_empty() {
            match self.decoder.read_frame()? {
                Some(frame) => {
                    self.buf.resize(frame.bytes_len(), 0);
                    frame.fill_buf::<E>(self.buf.make_contiguous());
                    self.buf.read(buf)
                }
                None => {
                    return Ok(0);
                }
            }
        } else {
            self.buf.read(buf)
        }
    }
}

impl<R: std::io::Read, E: crate::audio::Endianness> std::io::BufRead for Reader<R, E> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.buf.is_empty() {
            match self.decoder.read_frame()? {
                Some(frame) => {
                    self.buf.resize(frame.bytes_len(), 0);
                    frame.fill_buf::<E>(self.buf.make_contiguous());
                    self.buf.fill_buf()
                }
                None => {
                    return Ok(&[]);
                }
            }
        } else {
            self.buf.fill_buf()
        }
    }

    fn consume(&mut self, amt: usize) {
        self.buf.consume(amt)
    }
}

/// A FLAC decoder
pub struct Decoder<R> {
    reader: R,
    streaminfo: Streaminfo,
    seektable: Option<SeekTable>,
    // the size of everything before the first frame, in bytes
    frames_start: u64,
    samples_remaining: Option<u64>,
    buf: Frame,
}

impl<R: std::io::Read> Decoder<R> {
    /// Builds a new FLAC decoder from the given stream
    ///
    /// This assumes the stream is positioned at the start
    /// of the file.
    ///
    /// # Errors
    ///
    /// Returns an error of the initial FLAC metadata
    /// is invalid or an I/O error occurs reading
    /// the initial metadata.
    pub fn new(mut reader: R) -> Result<Self, Error> {
        use crate::Counter;
        use crate::metadata::{Block, read_blocks};
        use std::io::Read;

        let mut streaminfo = None;
        let mut seektable = None;
        let mut counter = Counter::new(reader.by_ref());

        for block in read_blocks(counter.by_ref()) {
            match block? {
                Block::Streaminfo(block) => {
                    streaminfo = Some(block);
                }
                Block::SeekTable(block) => {
                    seektable = Some(block);
                }
                _ => { /* ignore other blocks */ }
            }
        }

        match streaminfo {
            Some(streaminfo) => Ok(Self {
                frames_start: counter.count,
                reader,
                samples_remaining: streaminfo.total_samples.map(|s| s.get()),
                streaminfo,
                seektable,
                buf: Frame::default(),
            }),
            // read_blocks should check for this already
            // but we'll add a second check to be certain
            None => Err(Error::MissingStreaminfo),
        }
    }

    /// Returns channel count
    ///
    /// From 1 to 8
    pub fn channel_count(&self) -> NonZero<u8> {
        self.streaminfo.channels
    }

    /// Returns sample rate, in Hz
    pub fn sample_rate(&self) -> u32 {
        self.streaminfo.sample_rate
    }

    /// Returns decoder's bits-per-sample
    ///
    /// From 1 to 32
    pub fn bits_per_sample(&self) -> SignedBitCount<32> {
        self.streaminfo.bits_per_sample
    }

    /// Returns total number of channel-independent samples, if known
    pub fn total_samples(&self) -> Option<NonZero<u64>> {
        self.streaminfo.total_samples
    }

    /// Returns MD5 of entire stream, if known
    pub fn md5(&self) -> Option<&[u8; 16]> {
        self.streaminfo.md5.as_ref()
    }

    /// Returns decoded frame, if any.
    ///
    /// # Errors
    ///
    /// Returns any decoding error from the stream.
    pub fn read_frame(&mut self) -> Result<Option<&Frame>, Error> {
        use crate::crc::{Checksum, Crc16, CrcReader};
        use crate::stream::{ChannelAssignment, FrameHeader};
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc16_reader: CrcReader<_, Crc16> = CrcReader::new(self.reader.by_ref());

        let header = match self.samples_remaining {
            Some(0) => return Ok(None),
            Some(remaining) => FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo)
                .and_then(|header| {
                    // only the last block in a stream may contain <= 14 samples
                    let block_size = u16::from(header.block_size);
                    (u64::from(block_size) == remaining || block_size > 14)
                        .then_some(header)
                        .ok_or(Error::ShortBlock)
                })?,
            // if total number of remaining samples isn't known,
            // treat an EOF error as the end of stream
            // (this is an uncommon case)
            None => match FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo) {
                Ok(header) => header,
                Err(Error::Io(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(None);
                }
                Err(err) => return Err(err),
            },
        };

        let mut reader = BitReader::endian(crc16_reader.by_ref(), BigEndian);
        let buf = &mut self.buf;

        match header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => {
                buf.resized_channels(
                    header.sample_rate.into(),
                    header.bits_per_sample.into(),
                    total_channels.into(),
                    u16::from(header.block_size).into(),
                )
                .try_for_each(|channel| {
                    read_subframe(&mut reader, header.bits_per_sample, channel)
                })?;
            }
            ChannelAssignment::LeftSide => {
                let (left, side) = buf.resized_stereo(
                    header.sample_rate.into(),
                    header.bits_per_sample.into(),
                    u16::from(header.block_size).into(),
                );

                read_subframe(&mut reader, header.bits_per_sample, left)?;

                read_subframe(
                    &mut reader,
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                    side,
                )?;

                left.iter().zip(side.iter_mut()).for_each(|(left, side)| {
                    *side = *left - *side;
                });
            }
            ChannelAssignment::SideRight => {
                let (side, right) = buf.resized_stereo(
                    header.sample_rate.into(),
                    header.bits_per_sample.into(),
                    u16::from(header.block_size).into(),
                );

                read_subframe(
                    &mut reader,
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                    side,
                )?;

                read_subframe(&mut reader, header.bits_per_sample, right)?;

                side.iter_mut().zip(right.iter()).for_each(|(side, right)| {
                    *side += *right;
                });
            }
            ChannelAssignment::MidSide => {
                let (mid, side) = buf.resized_stereo(
                    header.sample_rate.into(),
                    header.bits_per_sample.into(),
                    u16::from(header.block_size).into(),
                );

                read_subframe(&mut reader, header.bits_per_sample, mid)?;

                read_subframe(
                    &mut reader,
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                    side,
                )?;

                mid.iter_mut().zip(side.iter_mut()).for_each(|(mid, side)| {
                    let sum = *mid * 2 + side.abs() % 2;
                    *mid = (sum + *side) >> 1;
                    *side = (sum - *side) >> 1;
                });
            }
        }

        reader.byte_align();
        reader.skip(16)?; // CRC-16 checksum

        if !crc16_reader.into_checksum().valid() {
            return Err(Error::Crc16Mismatch);
        }

        if let Some(remaining) = self.samples_remaining.as_mut() {
            *remaining = remaining
                .checked_sub(u16::from(header.block_size) as u64)
                .ok_or(Error::TooManySamples)?;
        }

        Ok(Some(buf))
    }
}

impl<R: std::io::Seek> Decoder<R> {
    /// Attempts to seek to desired sample number
    ///
    /// Upon success, returns the actual sample number
    /// the stream is positioned to, which may be less
    /// than the desired sample.
    ///
    /// # Errors
    ///
    /// Passes along an I/O error that occurs when seeking
    /// within the file.
    pub fn seek(&mut self, sample: u64) -> Result<u64, Error> {
        use crate::metadata::SeekPoint;
        use std::io::SeekFrom;

        match &self.seektable {
            Some(SeekTable { points: seektable }) => {
                match seektable
                    .iter()
                    .filter(|point| point.sample_offset.unwrap_or(u64::MAX) <= sample)
                    .next_back()
                {
                    Some(SeekPoint {
                        sample_offset: Some(sample_offset),
                        byte_offset,
                        ..
                    }) => {
                        assert!(*sample_offset <= sample);
                        self.reader
                            .seek(SeekFrom::Start(self.frames_start + byte_offset))?;
                        Ok(*sample_offset)
                    }
                    _ => {
                        // empty seektable so rewind to start of stream
                        self.reader.seek(SeekFrom::Start(self.frames_start))?;
                        Ok(0)
                    }
                }
            }
            None => {
                // no seektable
                // all we can do is rewind data to start of stream
                self.reader.seek(SeekFrom::Start(self.frames_start))?;
                Ok(0)
            }
        }
    }
}

fn read_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: SignedBitCount<32>,
    channel: &mut [i32],
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    let header = reader.parse::<SubframeHeader>()?;

    let effective_bps = bits_per_sample
        .checked_sub::<32>(header.wasted_bps)
        .ok_or(Error::ExcessiveWastedBits)?;

    match header.type_ {
        SubframeHeaderType::Constant => {
            channel.fill(reader.read_signed_counted(effective_bps)?);
        }
        SubframeHeaderType::Verbatim => {
            channel.iter_mut().try_for_each(|i| {
                *i = reader.read_signed_counted(effective_bps)?;
                Ok::<(), Error>(())
            })?;
        }
        SubframeHeaderType::Fixed { order } => {
            read_fixed_subframe(
                reader,
                effective_bps,
                SubframeHeaderType::FIXED_COEFFS[order as usize],
                channel,
            )?;
        }
        SubframeHeaderType::Lpc { order } => {
            read_lpc_subframe(reader, effective_bps, order, channel)?;
        }
    }

    if header.wasted_bps > 0 {
        channel.iter_mut().for_each(|i| *i <<= header.wasted_bps);
    }

    Ok(())
}

fn read_fixed_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: SignedBitCount<32>,
    coefficients: &[i64],
    channel: &mut [i32],
) -> Result<(), Error> {
    let (warm_up, residuals) = channel
        .split_at_mut_checked(coefficients.len())
        .ok_or(Error::InvalidFixedOrder)?;

    warm_up.iter_mut().try_for_each(|s| {
        *s = reader.read_signed_counted(bits_per_sample)?;
        Ok::<_, std::io::Error>(())
    })?;

    read_residuals(reader, coefficients.len(), residuals)?;
    predict(coefficients, 0, channel);
    Ok(())
}

fn read_lpc_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: SignedBitCount<32>,
    predictor_order: NonZero<u8>,
    channel: &mut [i32],
) -> Result<(), Error> {
    let mut coefficients: [i64; 32] = [0; 32];

    let (warm_up, residuals) = channel
        .split_at_mut_checked(predictor_order.get().into())
        .ok_or(Error::InvalidLpcOrder)?;

    warm_up.iter_mut().try_for_each(|s| {
        *s = reader.read_signed_counted(bits_per_sample)?;
        Ok::<_, std::io::Error>(())
    })?;

    let qlp_precision: SignedBitCount<15> = reader
        .read_count::<0b1111>()?
        .checked_add(1)
        .and_then(|c| c.signed_count())
        .ok_or(Error::InvalidQlpPrecision)?;

    let qlp_shift: u32 = reader
        .read::<5, i32>()?
        .try_into()
        .map_err(|_| Error::NegativeLpcShift)?;

    let coefficients = &mut coefficients[0..predictor_order.get().into()];

    coefficients.iter_mut().try_for_each(|c| {
        *c = reader.read_signed_counted(qlp_precision)?;
        Ok::<_, std::io::Error>(())
    })?;

    read_residuals(reader, coefficients.len(), residuals)?;
    predict(coefficients, qlp_shift, channel);
    Ok(())
}

fn predict(coefficients: &[i64], qlp_shift: u32, channel: &mut [i32]) {
    for split in coefficients.len()..channel.len() {
        let (predicted, residuals) = channel.split_at_mut(split);

        residuals[0] += (predicted
            .iter()
            .rev()
            .zip(coefficients)
            .map(|(x, y)| *x as i64 * y)
            .sum::<i64>()
            >> qlp_shift) as i32;
    }
}

#[test]
fn verify_prediction() {
    let mut coefficients = [-75, 166, 121, -269, -75, -399, 1042];
    let mut buffer = [
        -796, -547, -285, -32, 199, 443, 670, -2, -23, 14, 6, 3, -4, 12, -2, 10,
    ];
    coefficients.reverse();
    predict(&coefficients, 9, &mut buffer);
    assert_eq!(
        &buffer,
        &[
            -796, -547, -285, -32, 199, 443, 670, 875, 1046, 1208, 1343, 1454, 1541, 1616, 1663,
            1701
        ]
    );

    let mut coefficients = [119, -255, 555, -836, 879, -1199, 1757];
    let mut buffer = [-21363, -21951, -22649, -24364, -27297, -26870, -30017, 3157];
    coefficients.reverse();
    predict(&coefficients, 10, &mut buffer);
    assert_eq!(
        &buffer,
        &[
            -21363, -21951, -22649, -24364, -27297, -26870, -30017, -29718
        ]
    );

    let mut coefficients = [
        709, -2589, 4600, -4612, 1350, 4220, -9743, 12671, -12129, 8586, -3775, -645, 3904, -5543,
        4373, 182, -6873, 13265, -15417, 11550,
    ];
    let mut buffer = [
        213238, 210830, 234493, 209515, 235139, 201836, 208151, 186277, 157720, 148176, 115037,
        104836, 60794, 54523, 412, 17943, -6025, -3713, 8373, 11764, 30094,
    ];
    coefficients.reverse();
    predict(&coefficients, 12, &mut buffer);
    assert_eq!(
        &buffer,
        &[
            213238, 210830, 234493, 209515, 235139, 201836, 208151, 186277, 157720, 148176, 115037,
            104836, 60794, 54523, 412, 17943, -6025, -3713, 8373, 11764, 33931,
        ]
    );
}

fn read_residuals<R: BitRead>(
    reader: &mut R,
    predictor_order: usize,
    residuals: &mut [i32],
) -> Result<(), Error> {
    fn read_block<const RICE_MAX: u32, R: BitRead>(
        reader: &mut R,
        predictor_order: usize,
        mut residuals: &mut [i32],
    ) -> Result<(), Error> {
        use crate::stream::ResidualPartitionHeader;

        let block_size = predictor_order + residuals.len();
        let partition_order = reader.read::<4, u32>()?;
        let partition_count = 1 << partition_order;

        for p in 0..partition_count {
            let partition_size = (block_size / partition_count)
                .checked_sub(if p == 0 { predictor_order } else { 0 })
                .ok_or(Error::InvalidPartitionOrder)?;

            let (partition, next) = residuals
                .split_at_mut_checked(partition_size)
                .ok_or(Error::InvalidPartitionOrder)?;

            match reader.parse()? {
                ResidualPartitionHeader::Standard { rice } => {
                    partition.iter_mut().try_for_each(|s| {
                        let msb = reader.read_unary::<1>()?;
                        let lsb = reader.read_counted::<RICE_MAX, u32>(rice)?;
                        let unsigned = msb << u32::from(rice) | lsb;
                        *s = if (unsigned & 1) == 1 {
                            -((unsigned >> 1) as i32) - 1
                        } else {
                            (unsigned >> 1) as i32
                        };
                        Ok::<(), std::io::Error>(())
                    })?;
                }
                ResidualPartitionHeader::Escaped { escape_size } => {
                    partition.iter_mut().try_for_each(|s| {
                        *s = reader.read_signed_counted(escape_size)?;
                        Ok::<(), std::io::Error>(())
                    })?;
                }
                ResidualPartitionHeader::Constant => {
                    partition.fill(0);
                }
            }

            residuals = next;
        }

        Ok(())
    }

    match reader.read::<2, u8>()? {
        0 => read_block::<0b1111, R>(reader, predictor_order, residuals),
        1 => read_block::<0b11111, R>(reader, predictor_order, residuals),
        _ => Err(Error::InvalidCodingMethod),
    }
}
