// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For encoding PCM samples to FLAC files

use crate::Error;
use crate::audio::Frame;
use crate::metadata::{Streaminfo, write_blocks};
use crate::stream::FrameNumber;
use bitstream_io::{BitWrite, BitWriter, LittleEndian, SignedBitCount};
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
    options: EncodingOptions,
    streaminfo: Streaminfo,
    frame_number: FrameNumber,
    samples_written: u64,
    md5: BitWriter<md5::Context, LittleEndian>,
    finalized: bool,
}

impl<W: std::io::Write + std::io::Seek> Encoder<W> {
    const MAX_SAMPLES: u64 = 68_719_476_736;

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
    /// Note that if `total_samples` is indicated,
    /// the number written *must* be equal to that value
    /// or an error will occur when writing or finalizing the stream.
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
                    0..Self::MAX_SAMPLES => total_samples,
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
            options,
            streaminfo,
            frame_number: FrameNumber::default(),
            samples_written: 0,
            md5: BitWriter::new(md5::Context::new()),
            finalized: false,
        })
    }

    /// Encodes an audio frame of PCM samples
    ///
    /// Depending on the encoder's chosen block size,
    /// this may encode zero or more FLAC frames to disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error from the underlying stream,
    /// or if the frame's parameters are not a match
    /// for the encoder's.
    pub fn encode(&mut self, frame: &Frame) -> Result<(), Error> {
        // TODO - this would be a good candidate to replace with smallvec
        // since FLAC files are limited to 8 channels
        struct MultiIterator<I>(Vec<I>);

        impl<I: Iterator> Iterator for MultiIterator<I> {
            type Item = Vec<I::Item>;

            fn next(&mut self) -> Option<Self::Item> {
                let v = self
                    .0
                    .iter_mut()
                    .filter_map(|i| i.next())
                    .collect::<Vec<_>>();
                (!v.is_empty()).then_some(v)
            }
        }

        // sanity-check that frame's parameters match encoder's
        if frame.channel_count() != self.streaminfo.channels.get().into() {
            return Err(Error::ChannelsMismatch);
        } else if frame.bits_per_sample() != self.streaminfo.bits_per_sample.into() {
            return Err(Error::BitsPerSampleMismatch);
        } else if frame.sample_rate() != self.streaminfo.sample_rate {
            return Err(Error::SampleRateMismatch);
        }

        // update running total of samples written
        self.samples_written += frame.pcm_frames() as u64;
        if let Some(total_samples) = self.streaminfo.total_samples {
            if self.samples_written > total_samples.get() {
                return Err(Error::ExcessiveTotalSamples);
            }
        }

        // update MD5 calculation
        frame.iter().try_for_each(|i| {
            self.md5
                .write_signed_counted(self.streaminfo.bits_per_sample, i)?;
            self.md5.byte_align()
        })?;

        // TODO - partial frame must also be empty
        if frame.pcm_frames() % self.options.block_size as usize == 0 {
            let mut buffers = frame.channels().collect::<Vec<_>>();

            MultiIterator(
                buffers
                    .iter_mut()
                    .map(|b| b.chunks_exact(self.options.block_size as usize))
                    .collect(),
            )
            .try_for_each(|frame| self.encode_frame(frame))
        } else {
            // TODO - populate partial frames
            // TODO - encode any whole frames in partials
            // TODO - retain any remainder of partials
            todo!()
        }
    }

    fn encode_frame(&mut self, frame: Vec<&[i32]>) -> Result<(), Error> {
        use crate::Counter;
        use crate::crc::{Crc16, CrcWriter};
        use crate::stream::{ChannelAssignment, FrameHeader};
        use bitstream_io::BigEndian;

        debug_assert!(!frame.is_empty());

        let size = Counter::new(self.writer.by_ref());
        let mut w: CrcWriter<_, Crc16> = CrcWriter::new(size);

        // TODO - channel assignment may vary
        FrameHeader {
            blocking_strategy: false,
            frame_number: self.frame_number,
            block_size: frame[0].len() as u16,
            sample_rate: self.streaminfo.sample_rate,
            bits_per_sample: self.streaminfo.bits_per_sample,
            channel_assignment: ChannelAssignment::Independent(frame.len() as u8),
        }
        .write(&mut w, &self.streaminfo)?;

        let mut w = BitWriter::endian(w, BigEndian);

        for channel in frame {
            encode_subframe(w.by_ref(), channel, self.streaminfo.bits_per_sample)?;
        }

        let crc16: u16 = w.aligned_writer()?.checksum().into();
        w.write_from(crc16)?;

        self.frame_number.try_increment()?;

        // update minimum and maximum frame size values
        if let s @ Some(size) = u32::try_from(w.into_writer().into_writer().count)
            .ok()
            .filter(|size| *size < Streaminfo::MAX_FRAME_SIZE)
            .and_then(NonZero::new)
        {
            match &mut self.streaminfo.minimum_frame_size {
                Some(min_size) => {
                    *min_size = size.min(*min_size);
                }
                min_size @ None => {
                    *min_size = s;
                }
            }

            match &mut self.streaminfo.maximum_frame_size {
                Some(max_size) => {
                    *max_size = size.max(*max_size);
                }
                max_size @ None => {
                    *max_size = s;
                }
            }
        }

        Ok(())
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            self.finalized = true;

            // TODO - output any partial frame
            // TODO - update seektable

            match &mut self.streaminfo.total_samples {
                Some(expected) => {
                    if expected.get() != self.samples_written {
                        return Err(Error::SampleCountMismatch);
                    }
                }
                expected @ None => {
                    if self.samples_written < Self::MAX_SAMPLES {
                        *expected =
                            Some(NonZero::new(self.samples_written).ok_or(Error::NoSamples)?);
                    } else {
                        return Err(Error::ExcessiveTotalSamples);
                    }
                }
            }

            self.streaminfo.md5 = Some(self.md5.aligned_writer()?.clone().compute().0);

            self.writer.rewind()?;

            write_blocks(
                std::iter::once(&self.streaminfo.clone().into()),
                self.writer.by_ref(),
            )
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

fn encode_subframe<W: BitWrite>(
    mut w: W,
    channel: &[i32],
    bits_per_sample: SignedBitCount<32>,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    // TODO - try different subframe types
    // TODO - determine any wasted bits

    w.build(&SubframeHeader {
        type_: SubframeHeaderType::Verbatim,
        wasted_bps: 0,
    })?;

    channel
        .iter()
        .try_for_each(|i| w.write_signed_counted(bits_per_sample, *i))?;

    Ok(())
}
