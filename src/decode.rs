// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For decoding FLAC files to PCM samples

use crate::Error;
use crate::metadata::Streaminfo;
use bitstream_io::BitRead;

/// A FLAC decoder
pub struct Decoder<R> {
    reader: R,
    streaminfo: Streaminfo,
    buffer: Box<[i32]>,
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
                buffer: vec![
                    0;
                    usize::from(streaminfo.maximum_block_size)
                        * usize::from(streaminfo.channels.get())
                ]
                .into_boxed_slice(),
                reader,
                streaminfo,
            }),
            // read_blocks should check for this already
            // but we'll add a second check to be certain
            None => Err(Error::MissingStreaminfo),
        }
    }

    /// Reads a whole FLAC frame
    ///
    /// The frame may be empty at the end of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if an I/O error occurs when reading
    /// the stream, or if the stream data is invalid.
    pub fn read_frame(&mut self) -> Result<&[i32], Error> {
        use crate::crc::{Checksum, Crc16, CrcReader};
        use crate::stream::{ChannelAssignment, FrameHeader};
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc16_reader: CrcReader<_, Crc16> = CrcReader::new(self.reader.by_ref());
        let header = dbg!(FrameHeader::read(crc16_reader.by_ref(), &self.streaminfo)?);
        let channels = header.channel_assignment.len();
        let buffer = &mut self.buffer[0..usize::from(header.block_size) * usize::from(channels)];

        let mut reader = BitReader::endian(crc16_reader.by_ref(), BigEndian);

        match header.channel_assignment {
            ChannelAssignment::Independent(total_channels) => {
                (0..total_channels).try_for_each(|channel| {
                    read_subframe(
                        &mut reader,
                        header.bits_per_sample,
                        Stripe::new(buffer, channel, total_channels),
                    )
                })?;
            }
            _ => todo!(),
        }

        reader.byte_align();
        reader.skip(16)?; // CRC-16 checksum

        match crc16_reader.into_checksum().valid() {
            true => Ok(buffer),
            false => Err(Error::Crc16Mismatch),
        }
    }
}

fn read_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: u8,
    mut stripe: Stripe<'_>,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    let header: SubframeHeader = dbg!(reader.parse::<SubframeHeader>()?);

    let effective_bps = u32::from(bits_per_sample)
        .checked_sub(header.wasted_bps)
        .ok_or(Error::ExcessiveWastedBits)?;

    match header.type_ {
        SubframeHeaderType::Constant => {
            let sample = reader.read(effective_bps)?;
            stripe.for_each(|i| *i = sample);
        }
        SubframeHeaderType::Verbatim => {
            stripe.try_for_each(|i| {
                *i = reader.read(effective_bps)?;
                Ok::<(), Error>(())
            })?;
        }
        SubframeHeaderType::Fixed(predictor_order) => {
            read_fixed_subframe(reader, bits_per_sample, predictor_order, &mut stripe)?;
        }
        SubframeHeaderType::Lpc(_) => {
            todo!()
        }
    }

    if header.wasted_bps > 0 {
        stripe.for_each(|i| *i <<= header.wasted_bps);
    }

    Ok(())
}

fn read_fixed_subframe<R: BitRead>(
    reader: &mut R,
    bits_per_sample: u8,
    predictor_order: u8,
    channel: &mut Stripe<'_>,
) -> Result<(), Error> {
    match predictor_order {
        0 => {
            let mut residuals = ResidualBlock::new(reader, channel.len(), predictor_order)?;

            channel.try_for_each(|i| {
                *i = residuals.next()?;
                Ok(())
            })
        }
        1 => {
            // warm-up samples
            for i in 0..1 {
                channel[i] = reader.read(bits_per_sample.into())?;
            }
            let mut residuals = ResidualBlock::new(reader, channel.len(), predictor_order)?;
            for i in 1..channel.len() {
                channel[i] = channel[i - 1] + residuals.next()?;
            }
            Ok(())
        }
        2 => {
            // warm-up samples
            for i in 0..2 {
                channel[i] = reader.read(bits_per_sample.into())?;
            }
            let mut residuals = ResidualBlock::new(reader, channel.len(), predictor_order)?;
            for i in 2..channel.len() {
                channel[i] = (2 * channel[i - 1]) - channel[i - 2] + residuals.next()?;
            }
            Ok(())
        }
        3 => {
            // warm-up sampels
            for i in 0..3 {
                channel[i] = reader.read(bits_per_sample.into())?;
            }
            let mut residuals = ResidualBlock::new(reader, channel.len(), predictor_order)?;
            for i in 3..channel.len() {
                channel[i] = (3 * channel[i - 1]) - (3 * channel[i - 2])
                    + channel[i - 3]
                    + residuals.next()?;
            }
            Ok(())
        }
        4 => {
            for i in 0..4 {
                channel[i] = reader.read(bits_per_sample.into())?;
            }
            let mut residuals = ResidualBlock::new(reader, channel.len(), predictor_order)?;
            for i in 4..channel.len() {
                channel[i] = (4 * channel[i - 1]) - (6 * channel[i - 2]) + (4 * channel[i - 3])
                    - channel[i - 4]
                    + residuals.next()?;
            }
            Ok(())
        }
        // this shouldn't happen
        _ => panic!("invalid FIXED subframe predictor order"),
    }
}

struct ResidualBlock<'r, R: BitRead> {
    reader: &'r mut R,
    block_size: u16,
    rice_bits: u32,
    escaped_rice: u32,

    partitions: std::ops::Range<u16>,
    partition: ResidualPartition,
}

impl<'r, R: BitRead> ResidualBlock<'r, R> {
    fn new(reader: &'r mut R, block_size: u16, predictor_order: u8) -> Result<Self, Error> {
        let coding_method = reader.read_in::<2, u8>()?;
        let partition_order = reader.read_in::<4, u32>()?;
        let partition_count = 1 << partition_order;
        let (rice_bits, escaped_rice) = match coding_method {
            0b00 => (4, 0b1111),
            0b01 => (5, 0b11111),
            _ => return Err(Error::InvalidCodingMethod),
        };

        if ((block_size % partition_count) != 0)
            || (u16::from(predictor_order) > (block_size / partition_count))
        {
            return Err(Error::InvalidPartitionOrder);
        }

        Ok(Self {
            block_size,
            rice_bits,
            escaped_rice,

            partitions: 0..partition_count,
            partition: ResidualPartition::new(
                reader,
                rice_bits,
                escaped_rice,
                block_size / partition_count - u16::from(predictor_order),
            )?,

            reader,
        })
    }
}

impl<R: BitRead> ResidualBlock<'_, R> {
    fn next(&mut self) -> Result<i32, Error> {
        match self.partition.next(self.reader) {
            Some(residual) => residual.map_err(Error::Io),
            None => {
                self.partition = match self.partitions.next() {
                    Some(_) => ResidualPartition::new(
                        self.reader,
                        self.rice_bits,
                        self.escaped_rice,
                        self.block_size / self.partitions.end,
                    )?,
                    // this shouldn't happen?
                    None => panic!("attempting to read too many partitions"),
                };
                self.next()
            }
        }
    }
}

enum ResidualPartition {
    Unescaped {
        rice: u32,
        residual: std::ops::Range<u16>,
    },
    Escaped {
        escape_code: u32,
        residual: std::ops::Range<u16>,
    },
}

impl ResidualPartition {
    fn new<R: BitRead>(
        r: &mut R,
        rice_bits: u32,
        escaped_rice: u32,
        partition_size: u16,
    ) -> Result<Self, Error> {
        let rice = r.read(rice_bits)?;
        Ok(if rice == escaped_rice {
            Self::Escaped {
                escape_code: r.read(5)?,
                residual: 0..partition_size,
            }
        } else {
            Self::Unescaped {
                rice,
                residual: 0..partition_size,
            }
        })
    }
}

impl ResidualPartition {
    fn next<R: BitRead>(&mut self, r: &mut R) -> Option<Result<i32, std::io::Error>> {
        match self {
            Self::Unescaped { rice, residual } => residual.next().map(|_| {
                let msb = r.read_unary::<1>()?;
                let lsb = r.read::<u32>(*rice)?;
                let unsigned = (msb << *rice) | lsb;
                Ok(if unsigned % 2 == 1 {
                    -((unsigned >> 1) as i32) - 1
                } else {
                    (unsigned >> 1) as i32
                })
            }),
            Self::Escaped {
                escape_code,
                residual,
            } => residual.next().map(|_| r.read(*escape_code)),
        }
    }
}

struct Stripe<'b> {
    buf: &'b mut [i32],
    channel: u8,
    total_channels: u8,
}

impl<'b> Stripe<'b> {
    fn new(buf: &'b mut [i32], channel: u8, total_channels: u8) -> Self {
        Self {
            buf,
            channel,
            total_channels,
        }
    }

    fn len(&self) -> u16 {
        (self.buf.len() / usize::from(self.total_channels))
            .try_into()
            .unwrap()
    }

    fn for_each(&mut self, mut f: impl FnMut(&mut i32)) {
        for i in 0..self.len() {
            f(&mut self[i]);
        }
    }

    fn try_for_each<E>(&mut self, mut f: impl FnMut(&mut i32) -> Result<(), E>) -> Result<(), E> {
        for i in 0..self.len() {
            f(&mut self[i])?;
        }
        Ok(())
    }
}

impl std::ops::Index<u16> for Stripe<'_> {
    type Output = i32;

    fn index(&self, index: u16) -> &i32 {
        &self.buf[usize::from(index) * usize::from(self.total_channels) + usize::from(self.channel)]
    }
}

impl std::ops::IndexMut<u16> for Stripe<'_> {
    fn index_mut(&mut self, index: u16) -> &mut i32 {
        &mut self.buf
            [usize::from(index) * usize::from(self.total_channels) + usize::from(self.channel)]
    }
}
