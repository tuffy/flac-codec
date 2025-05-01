// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For encoding PCM samples to FLAC files

use crate::Error;
use crate::metadata::{Streaminfo, write_blocks};
use bitstream_io::SignedBitCount;
use std::num::NonZero;

/// FLAC encoding options
pub struct EncodingOptions {
    block_size: u16,
}

impl EncodingOptions {
    /// Assigns new block size to options
    pub fn block_size(self, block_size: u16) -> Self {
        Self { block_size }
    }
}

impl Default for EncodingOptions {
    fn default() -> Self {
        Self { block_size: 4096 }
    }
}

/// A FLAC encoder
pub struct Encoder<W: std::io::Write + std::io::Seek> {
    writer: W,
    streaminfo: Streaminfo,
    finalized: bool,
}

impl<W: std::io::Write + std::io::Seek> Encoder<W> {
    /// Creates new encoder with the given parameters
    ///
    /// `sample_rate` must be between 0 (for non-audio streams)
    /// and 1048576 (a 20 bit field).
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// `total_samples`, if known, must be between
    /// 1 and 68_719_476_736 (a 36 bit field).
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    /// Returns error if any of the encoding parameters are invalid.
    pub fn new(
        mut writer: W,
        options: EncodingOptions,
        sample_rate: u32,
        bits_per_sample: impl TryInto<SignedBitCount<32>>,
        channels: NonZero<u8>,
        total_samples: Option<NonZero<u64>>,
    ) -> Result<Self, Error> {
        let streaminfo = Streaminfo {
            minimum_block_size: options.block_size,
            maximum_block_size: options.block_size,
            minimum_frame_size: None,
            maximum_frame_size: None,
            sample_rate: (0..1048576)
                .contains(&sample_rate)
                .then_some(sample_rate)
                .ok_or(Error::InvalidSampleRate)?,
            bits_per_sample: bits_per_sample
                .try_into()
                .map_err(|_| Error::InvalidBitsPerSample)?,
            channels: (0..=8)
                .contains(&channels.get())
                .then_some(channels)
                .ok_or(Error::ExcessiveChannels)?,
            total_samples: match total_samples {
                None => None,
                total_samples @ Some(samples) => match samples.get() {
                    0..68_719_476_736 => total_samples,
                    _ => return Err(Error::ExcessiveTotalSamples),
                },
            },
            md5: None,
        };

        // TODO - include SEEKTABLE block
        // TODO - include PADDING block, if requested

        write_blocks(std::iter::once(&streaminfo.clone().into()), writer.by_ref())?;

        Ok(Self {
            writer,
            streaminfo,
            finalized: false,
        })
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            // TODO - output any unwritten samples
            // TODO - ensure total written samples are what's expected
            // TODO - update seektable
            // TODO - finalize stream MD5

            self.writer.rewind()?;

            write_blocks(
                std::iter::once(&self.streaminfo.clone().into()),
                self.writer.by_ref(),
            )
            .inspect(|_| {
                self.finalized = true;
            })
            .inspect_err(|_| {
                self.finalized = true;
            })
        } else {
            Ok(())
        }
    }

    /// Attempt to finalize stream
    ///
    /// It is necessary to finalize the FLAC encoder
    /// so that it will write any partially unwritten samples
    /// to the stream and update the STREAMINFO and SEEKTABLE blocks
    /// with their final values.
    ///
    /// Dropping the encoder will attempt to finalize the stream
    /// automatically, but will ignore any errors that may occur.
    pub fn finalize(mut self) -> Result<(), Error> {
        self.finalize_inner()?;
        Ok(())
    }
}

impl<W: std::io::Write + std::io::Seek> Drop for Encoder<W> {
    fn drop(&mut self) {
        let _ = self.finalize_inner();
    }
}
