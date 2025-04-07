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
use bitstream_io::{BitCount, BitRead};
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
        use crate::metadata::{Block, read_blocks};

        let mut streaminfo = None;

        for block in read_blocks(reader.by_ref()) {
            match block? {
                Block::Streaminfo(s) => {
                    streaminfo = Some(s);
                }
                // FIXME - get SEEKTABLE for file seeking
                // FIXME - get VORBIS_COMMENT for channel mask
                _ => { /* ignore other blocks */ }
            }
        }

        match streaminfo {
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
            Some(0) => return Ok(buf.empty(&self.streaminfo)),
            Some(_) => FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo)?,
            // if total number of remaining samples isn't known,
            // treat an EOF error as the end of stream
            // (this is an uncommon case)
            None => match FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo) {
                Ok(header) => header,
                Err(Error::Io(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(buf.empty(&self.streaminfo));
                }
                Err(err) => return Err(err),
            },
        };

        let mut reader = BitReader::endian(crc16_reader.by_ref(), BigEndian);

        match header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => {
                buf.channels.resize_with(total_channels.into(), || vec![]);
                buf.channels.iter_mut().try_for_each(|channel| {
                    channel.resize(header.block_size.into(), 0);
                    read_subframe(&mut reader, header.bits_per_sample, channel)
                })?;
            }
            ChannelAssignment::LeftSide => {
                buf.channels.resize_with(2, || vec![]);
                let [left, side] = buf.channels.get_disjoint_mut([0, 1]).unwrap();

                left.resize(header.block_size.into(), 0);
                read_subframe(&mut reader, header.bits_per_sample, left)?;

                side.resize(header.block_size.into(), 0);
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
                buf.channels.resize_with(2, || vec![]);
                let [side, right] = buf.channels.get_disjoint_mut([0, 1]).unwrap();

                side.resize(header.block_size.into(), 0);
                read_subframe(
                    &mut reader,
                    header
                        .bits_per_sample
                        .checked_add(1)
                        .ok_or(Error::ExcessiveBps)?,
                    side,
                )?;

                right.resize(header.block_size.into(), 0);
                read_subframe(&mut reader, header.bits_per_sample, right)?;

                side.iter_mut().zip(right.iter()).for_each(|(side, right)| {
                    *side = *side + *right;
                });
            }
            ChannelAssignment::MidSide => {
                buf.channels.resize_with(2, || vec![]);
                let [mid, side] = buf.channels.get_disjoint_mut([0, 1]).unwrap();

                mid.resize(header.block_size.into(), 0);
                read_subframe(&mut reader, header.bits_per_sample, mid)?;

                side.resize(header.block_size.into(), 0);
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
                buf.bits_per_sample = u32::from(header.bits_per_sample) as u8;
                buf.sample_rate = header.sample_rate;
                Ok(buf)
            }
            false => Err(Error::Crc16Mismatch),
        }
    }
}

fn read_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: BitCount<32>,
    channel: &mut [i32],
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    let header = reader.parse::<SubframeHeader>()?;

    let effective_bps = bits_per_sample
        .checked_sub::<32>(header.wasted_bps)
        .ok_or(Error::ExcessiveWastedBits)?;

    match header.type_ {
        SubframeHeaderType::Constant => {
            let sample = reader.read_counted(effective_bps)?;
            channel.iter_mut().for_each(|i| *i = sample);
        }
        SubframeHeaderType::Verbatim => {
            channel.iter_mut().try_for_each(|i| {
                *i = reader.read_counted(effective_bps)?;
                Ok::<(), Error>(())
            })?;
        }
        SubframeHeaderType::Fixed(predictor_order) => {
            read_fixed_subframe(reader, bits_per_sample, predictor_order, channel)?;
        }
        SubframeHeaderType::Lpc(predictor_order) => {
            read_lpc_subframe(reader, bits_per_sample, predictor_order, channel)?;
        }
    }

    if header.wasted_bps > 0 {
        channel.iter_mut().for_each(|i| *i <<= header.wasted_bps);
    }

    Ok(())
}

fn read_fixed_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: BitCount<32>,
    predictor_order: u8,
    channel: &mut [i32],
) -> Result<(), Error> {
    let (warm_up, residuals) = channel
        .split_at_mut_checked(predictor_order.into())
        .ok_or_else(|| Error::InvalidFixedOrder)?;

    warm_up.iter_mut().try_for_each(|s| {
        *s = reader.read_counted(bits_per_sample)?;
        Ok::<_, std::io::Error>(())
    })?;

    read_residuals(reader, predictor_order, residuals)?;

    predict_fixed(predictor_order, channel)
}

fn predict_fixed(predictor_order: u8, channel: &mut [i32]) -> Result<(), Error> {
    let coefficients: &[i32] = match predictor_order {
        0 => &[],
        1 => &[1],
        2 => &[2, -1],
        3 => &[3, -3, 1],
        4 => &[4, -6, 4, -1],
        _ => return Err(Error::InvalidFixedOrder),
    };

    for split in predictor_order.into()..channel.len() {
        let (predicted, residuals) = channel.split_at_mut(split);

        residuals[0] += predicted
            .iter()
            .rev()
            .zip(coefficients)
            .map(|(x, y)| x * y)
            .sum::<i32>();
    }

    Ok(())
}

#[test]
fn test_fixed_prediction() {
    let mut buffer = [
        -729, -722, -667, -19, -16, 17, -23, -7, 16, -16, -5, 3, -8, -13, -15, -1,
    ];
    assert!(predict_fixed(3, &mut buffer).is_ok());
    assert_eq!(
        &buffer,
        &[
            -729, -722, -667, -583, -486, -359, -225, -91, 59, 209, 354, 497, 630, 740, 812, 845
        ]
    );
}

fn read_lpc_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: BitCount<32>,
    predictor_order: NonZero<u8>,
    channel: &mut [i32],
) -> Result<(), Error> {
    let mut coefficients: [i32; 32] = [0; 32];

    let (warm_up, residuals) = channel
        .split_at_mut_checked(predictor_order.get().into())
        .ok_or_else(|| Error::InvalidLpcOrder)?;

    warm_up.iter_mut().try_for_each(|s| {
        *s = reader.read_counted(bits_per_sample)?;
        Ok::<_, std::io::Error>(())
    })?;

    let qlp_precision: BitCount<16> = reader.read_count::<0b1111>()?.checked_add(1).unwrap();

    let qlp_shift: u32 = reader
        .read::<5, i32>()?
        .try_into()
        .map_err(|_| Error::NegativeLpcShift)?;

    let coefficients = &mut coefficients[0..predictor_order.get().into()];

    coefficients.iter_mut().try_for_each(|c| {
        *c = reader.read_counted(qlp_precision)?;
        Ok::<_, std::io::Error>(())
    })?;

    read_residuals(reader, predictor_order.get(), residuals)?;

    Ok(predict_lpc(coefficients, qlp_shift, channel))
}

fn predict_lpc(coefficients: &[i32], qlp_shift: u32, channel: &mut [i32]) {
    for split in coefficients.len()..channel.len() {
        let (predicted, residuals) = channel.split_at_mut(split);

        residuals[0] += (predicted
            .iter()
            .rev()
            .zip(coefficients)
            .map(|(x, y)| *x as i64 * *y as i64)
            .sum::<i64>()
            >> qlp_shift) as i32;
    }
}

#[test]
fn verify_lpc_prediction() {
    let mut coefficients = [-75, 166, 121, -269, -75, -399, 1042];
    let mut buffer = [
        -796, -547, -285, -32, 199, 443, 670, -2, -23, 14, 6, 3, -4, 12, -2, 10,
    ];
    coefficients.reverse();
    predict_lpc(&coefficients, 9, &mut buffer);
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
    predict_lpc(&coefficients, 10, &mut buffer);
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
    predict_lpc(&coefficients, 12, &mut buffer);
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
    predictor_order: u8,
    residuals: &mut [i32],
) -> Result<(), Error> {
    fn read_block<const RICE_MAX: u32, R: BitRead>(
        reader: &mut R,
        predictor_order: u8,
        mut residuals: &mut [i32],
    ) -> Result<(), Error> {
        let block_size = (predictor_order as usize) + residuals.len();
        let partition_order = reader.read::<4, u32>()?;
        let partition_count = 1 << partition_order;

        for p in 0..partition_count {
            let partition_size =
                block_size / partition_count - if p == 0 { predictor_order as usize } else { 0 };

            let rice = reader.read_count::<RICE_MAX>()?;

            let (partition, next) = residuals
                .split_at_mut_checked(partition_size)
                .ok_or_else(|| Error::InvalidPartitionOrder)?;

            if u32::from(rice) == RICE_MAX {
                // escaped residuals
                let escape_size = reader.read_count::<0b11111>()?;

                partition.iter_mut().try_for_each(|s| {
                    *s = reader.read_counted(escape_size)?;
                    Ok::<(), std::io::Error>(())
                })?;
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
