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
        SubframeHeaderType::Fixed(_) => {
            todo!()
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
