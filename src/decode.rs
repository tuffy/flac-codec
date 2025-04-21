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
use crate::metadata::Streaminfo;
use bitstream_io::{BitCount, BitRead, SignedBitCount};
use std::num::NonZero;

/// A FLAC decoder
pub struct Decoder<R> {
    reader: R,
    streaminfo: Streaminfo,
    samples_remaining: Option<u64>,
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
        use crate::metadata::read_streaminfo;

        match read_streaminfo(reader.by_ref())? {
            Some(streaminfo) => Ok(Self {
                reader,
                samples_remaining: streaminfo.total_samples.map(|s| s.get()),
                streaminfo,
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

    /// Returns MD5 of entire stream, if known
    pub fn md5(&self) -> Option<&[u8; 16]> {
        self.streaminfo.md5.as_ref()
    }

    /// Given a frame buffer, returns a decoded frame.
    ///
    /// `Frame::default` may be used if no frame buffer
    /// exists to be reused.
    ///
    /// # Errors
    ///
    /// Returns any decoding error from the stream.
    pub fn read_frame(&mut self, mut buf: Frame) -> Result<Frame, Error> {
        use crate::crc::{Checksum, Crc16, CrcReader};
        use crate::stream::{ChannelAssignment, FrameHeader};
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc16_reader: CrcReader<_, Crc16> = CrcReader::new(self.reader.by_ref());

        let header = match self.samples_remaining {
            Some(0) => return Ok(buf.empty()),
            Some(remaining) => FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo)
                .and_then(|header| {
                    // only the last block in a stream may contain <= 14 samples
                    (u64::from(header.block_size) == remaining || header.block_size > 14)
                        .then_some(header)
                        .ok_or(Error::ShortBlock)
                })?,
            // if total number of remaining samples isn't known,
            // treat an EOF error as the end of stream
            // (this is an uncommon case)
            None => match FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo) {
                Ok(header) => header,
                Err(Error::Io(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(buf.empty());
                }
                Err(err) => return Err(err),
            },
        };

        let mut reader = BitReader::endian(crc16_reader.by_ref(), BigEndian);

        match header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => {
                buf.resize_for(
                    header.sample_rate,
                    header.bits_per_sample.into(),
                    total_channels.into(),
                    header.block_size.into(),
                )
                .try_for_each(|channel| {
                    read_subframe(&mut reader, header.bits_per_sample, channel)
                })?;
            }
            ChannelAssignment::LeftSide => {
                let (left, side) = buf.resize_for_2(
                    header.sample_rate,
                    header.bits_per_sample.into(),
                    header.block_size.into(),
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
                let (side, right) = buf.resize_for_2(
                    header.sample_rate,
                    header.bits_per_sample.into(),
                    header.block_size.into(),
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
                let (mid, side) = buf.resize_for_2(
                    header.sample_rate,
                    header.bits_per_sample.into(),
                    header.block_size.into(),
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

        match crc16_reader.into_checksum().valid() {
            true => {
                if let Some(remaining) = self.samples_remaining.as_mut() {
                    *remaining = remaining
                        .checked_sub(u64::from(header.block_size))
                        .ok_or(Error::TooManySamples)?;
                }
                Ok(buf)
            }
            false => Err(Error::Crc16Mismatch),
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
        let block_size = predictor_order + residuals.len();
        let partition_order = reader.read::<4, u32>()?;
        let partition_count = 1 << partition_order;

        for p in 0..partition_count {
            let partition_size = (block_size / partition_count)
                .checked_sub(if p == 0 { predictor_order } else { 0 })
                .ok_or(Error::InvalidPartitionOrder)?;

            let rice = reader.read_count::<RICE_MAX>()?;

            let (partition, next) = residuals
                .split_at_mut_checked(partition_size)
                .ok_or(Error::InvalidPartitionOrder)?;

            if rice == BitCount::new::<{ RICE_MAX }>() {
                // escaped residuals

                match reader.read_count::<0b11111>()?.signed_count() {
                    None => {
                        partition.fill(0);
                    }
                    Some(escape_size) => {
                        partition.iter_mut().try_for_each(|s| {
                            *s = reader.read_signed_counted(escape_size)?;
                            Ok::<(), std::io::Error>(())
                        })?;
                    }
                }
            } else {
                // regular residuals
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
