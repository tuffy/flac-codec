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
use bitstream_io::{BitRead, FromBitStreamWith};

#[derive(Debug)]
pub struct FrameHeader {
    blocking_strategy: bool,
    block_size: u16,
    sample_rate: u32,
    channel_assignment: ChannelAssignment,
    bits_per_sample: u8,
    frame_number: u32,
}

impl FromBitStreamWith<'_> for FrameHeader {
    type Error = Error;
    type Context = Streaminfo;

    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        streaminfo: &Streaminfo,
    ) -> Result<Self, Self::Error> {
        if r.read_in::<15, u16>()? != 0b111111111111100 {
            return Err(Error::InvalidSyncCode);
        }
        let blocking_strategy = r.read_bit()?;
        let encoded_block_size = r.read_in::<4, u8>()?;
        let encoded_sample_rate = r.read_in::<4, u8>()?;
        let encoded_channels = r.read_in::<4, u8>()?;
        let encoded_bps = r.read_in::<3, u8>()?;
        r.skip(1)?;
        let frame_number = read_frame_number(r)?;

        Ok(Self {
            blocking_strategy,
            frame_number,
            block_size: match encoded_block_size {
                0b0000 => return Err(Error::InvalidBlockSize),
                0b0001 => 192,
                v @ 0b0010..=0b0101 => 144 * (1 << v),
                0b0110 => r.read_in::<8, u16>()? + 1,
                0b0111 => r.read_in::<16, u16>()? + 1,
                v @ 0b1000..=0b1111 => 1 << v,
                _ => unreachable!(), // 4-bit field
            },
            sample_rate: match encoded_sample_rate {
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
                0b1100 => r.read_in::<8, u32>()? * 1000,
                0b1101 => r.read_in::<16, _>()?,
                0b1110 => r.read_in::<16, u32>()? * 10,
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
                0b000 => streaminfo.bits_per_sample.get(),
                0b001 => 8,
                0b010 => 12,
                0b011 => return Err(Error::InvalidBitsPerSample),
                0b100 => 16,
                0b101 => 20,
                0b110 => 24,
                0b111 => 32,
                _ => unreachable!(), // 3-bit field
            },
        })
    }
}

#[derive(Debug)]
enum ChannelAssignment {
    Independent(u8),
    LeftSide,
    SideRight,
    MidSide,
}

fn read_frame_number<R: BitRead + ?Sized>(r: &mut R) -> Result<u32, Error> {
    match r.read_unary::<0>()? {
        0 => Ok(r.read_in::<7, _>()?),
        1 => Err(Error::InvalidFrameNumber),
        bytes @ 2..=7 => {
            let mut frame = r.read(7 - bytes)?;
            for _ in 1..bytes {
                match r.read_in::<2, u8>()? {
                    0b10 => {
                        frame = frame << 6 | r.read_in::<6, u32>()?;
                    }
                    _ => return Err(Error::InvalidFrameNumber),
                }
            }
            Ok(frame)
        }
        _ => Err(Error::InvalidFrameNumber),
    }
}
