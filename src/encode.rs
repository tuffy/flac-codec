// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For encoding PCM samples to FLAC files

use crate::audio::Frame;
use crate::metadata::{
    Application, BlockList, BlockSize, Cuesheet, Picture, SeekPoint, Streaminfo, VorbisComment,
    write_blocks,
};
use crate::stream::{ChannelAssignment, FrameNumber, Independent, SampleRate};
use crate::{Counter, Error};
use arrayvec::ArrayVec;
use bitstream_io::{BigEndian, BitRecorder, BitWrite, BitWriter, SignedBitCount};
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::num::NonZero;
use std::path::Path;

const MAX_CHANNELS: usize = 8;

/// A FLAC writer which accepts samples as bytes
///
/// # Example
///
/// ```
/// use flac_codec::{
///     byteorder::LittleEndian,
///     encode::{FlacWriter, Options},
///     decode::{FlacReader, Metadata},
/// };
/// use std::io::{Cursor, Read, Seek, Write};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// let mut writer = FlacWriter::endian(
///     &mut flac,           // our wrapped writer
///     LittleEndian,        // .wav-style byte order
///     Options::default(),  // default encoding options
///     44100,               // sample rate
///     16,                  // bits-per-sample
///     1,                   // channel count
///     Some(2000),          // total bytes
/// ).unwrap();
///
/// // write 1000 samples as 16-bit, signed, little-endian bytes (2000 bytes total)
/// let written_bytes = (0..1000).map(i16::to_le_bytes).flatten().collect::<Vec<u8>>();
/// assert!(writer.write_all(&written_bytes).is_ok());
///
/// // finalize writing file
/// assert!(writer.finalize().is_ok());
///
/// flac.rewind().unwrap();
///
/// // open reader around written FLAC file
/// let mut reader = FlacReader::endian(flac, LittleEndian).unwrap();
///
/// // read 2000 bytes
/// let mut read_bytes = vec![];
/// assert!(reader.read_to_end(&mut read_bytes).is_ok());
///
/// // ensure MD5 sum of signed, little-endian samples matches hash in file
/// let mut md5 = md5::Context::new();
/// md5.consume(&read_bytes);
/// assert_eq!(&md5.compute().0, reader.md5().unwrap());
///
/// // ensure input and output matches
/// assert_eq!(read_bytes, written_bytes);
/// ```
pub struct FlacWriter<W: std::io::Write + std::io::Seek, E: crate::byteorder::Endianness> {
    // the wrapped encoder
    encoder: Encoder<W>,
    // bytes that make up a partial FLAC frame
    buf: VecDeque<u8>,
    // a whole set of samples for a FLAC frame
    frame: Frame,
    // size of a single sample in bytes
    bytes_per_sample: usize,
    // size of single set of channel-independent samples in bytes
    pcm_frame_size: usize,
    // size of whole FLAC frame's samples in bytes
    frame_byte_size: usize,
    // whether the encoder has finalized the file
    finalized: bool,
    // the input bytes' endianness
    endianness: std::marker::PhantomData<E>,
}

impl<W: std::io::Write + std::io::Seek, E: crate::byteorder::Endianness> FlacWriter<W, E> {
    /// Creates new FLAC writer with the given parameters
    ///
    /// The writer should be positioned at the start of the file.
    ///
    /// `sample_rate` must be between 0 (for non-audio streams) and 2²⁰.
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of channel-independent samples written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    /// Returns error if any of the encoding parameters are invalid.
    pub fn new(
        writer: W,
        options: Options,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u8,
        total_bytes: Option<u64>,
    ) -> Result<Self, Error> {
        let bits_per_sample: SignedBitCount<32> = bits_per_sample
            .try_into()
            .map_err(|_| Error::InvalidBitsPerSample)?;

        let bytes_per_sample = u32::from(bits_per_sample).div_ceil(8) as usize;

        let pcm_frame_size = bytes_per_sample * channels as usize;

        Ok(Self {
            buf: VecDeque::default(),
            frame: Frame::empty(channels.into(), bits_per_sample.into()),
            bytes_per_sample,
            pcm_frame_size,
            frame_byte_size: pcm_frame_size * options.block_size as usize,
            encoder: Encoder::new(
                writer,
                options,
                sample_rate,
                bits_per_sample,
                channels,
                total_bytes
                    .map(|bytes| {
                        exact_div(bytes, channels.into())
                            .and_then(|s| exact_div(s, bytes_per_sample as u64))
                            .ok_or(Error::SamplesNotDivisibleByChannels)
                            .and_then(|b| NonZero::new(b).ok_or(Error::InvalidTotalBytes))
                    })
                    .transpose()?,
            )?,
            finalized: false,
            endianness: std::marker::PhantomData,
        })
    }

    /// Creates new FLAC writer with CDDA parameters
    ///
    /// The writer should be positioned at the start of the file.
    ///
    /// Sample rate is 44100 Hz, bits-per-sample is 16,
    /// channels is 2.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of channel-independent samples written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    /// Returns error if any of the encoding parameters are invalid.
    pub fn new_cdda(writer: W, options: Options, total_bytes: Option<u64>) -> Result<Self, Error> {
        Self::new(writer, options, 44100, 16, 2, total_bytes)
    }

    /// Creates new FLAC writer in the given endianness with the given parameters
    ///
    /// The writer should be positioned at the start of the file.
    ///
    /// `sample_rate` must be between 0 (for non-audio streams) and 2²⁰.
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of bytes written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    #[inline]
    pub fn endian(
        writer: W,
        _endianness: E,
        options: Options,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u8,
        total_bytes: Option<u64>,
    ) -> Result<Self, Error> {
        Self::new(
            writer,
            options,
            sample_rate,
            bits_per_sample,
            channels,
            total_bytes,
        )
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            self.finalized = true;

            // encode as many bytes as possible into final frame, if necessary
            if !self.buf.is_empty() {
                use crate::byteorder::LittleEndian;

                // truncate buffer to whole PCM frames
                let buf = self.buf.make_contiguous();
                let buf_len = buf.len();
                let buf = &mut buf[..(buf_len - buf_len % self.pcm_frame_size)];

                // convert buffer to little-endian bytes
                E::bytes_to_le(buf, self.bytes_per_sample);

                // update MD5 sum with little-endian bytes
                self.encoder.update_md5(buf);

                self.encoder
                    .encode(self.frame.fill_from_buf::<LittleEndian>(buf))?;
            }

            self.encoder.finalize_inner()
        } else {
            Ok(())
        }
    }

    /// Attempt to finalize stream
    ///
    /// It is necessary to finalize the FLAC encoder
    /// so that it will write any partially unwritten samples
    /// to the stream and update the [`crate::metadata::Streaminfo`] and [`crate::metadata::SeekTable`] blocks
    /// with their final values.
    ///
    /// Dropping the encoder will attempt to finalize the stream
    /// automatically, but will ignore any errors that may occur.
    pub fn finalize(mut self) -> Result<(), Error> {
        self.finalize_inner()?;
        Ok(())
    }
}

impl<E: crate::byteorder::Endianness> FlacWriter<BufWriter<File>, E> {
    /// Creates new FLAC file at the given path
    ///
    /// `sample_rate` must be between 0 (for non-audio streams) and 2²⁰.
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of bytes written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    #[inline]
    pub fn create<P: AsRef<Path>>(
        path: P,
        options: Options,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u8,
        total_bytes: Option<u64>,
    ) -> Result<Self, Error> {
        FlacWriter::new(
            BufWriter::new(options.create(path)?),
            options,
            sample_rate,
            bits_per_sample,
            channels,
            total_bytes,
        )
    }

    /// Creates new FLAC file with CDDA parameters at the given path
    ///
    /// Sample rate is 44100 Hz, bits-per-sample is 16,
    /// channels is 2.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of bytes written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    pub fn create_cdda<P: AsRef<Path>>(
        path: P,
        options: Options,
        total_bytes: Option<u64>,
    ) -> Result<Self, Error> {
        Self::create(path, options, 44100, 16, 2, total_bytes)
    }
}

impl<W: std::io::Write + std::io::Seek, E: crate::byteorder::Endianness> std::io::Write
    for FlacWriter<W, E>
{
    /// Writes a set of sample bytes to the FLAC file
    ///
    /// Samples are encoded in the stream's given byte order.
    ///
    /// Samples are then interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        use crate::byteorder::LittleEndian;

        // dump whole set of bytes into our internal buffer
        self.buf.extend(buf);

        // encode as many FLAC frames as possible (which may be 0)
        let mut encoded_frames = 0;
        for buf in self
            .buf
            .make_contiguous()
            .chunks_exact_mut(self.frame_byte_size)
        {
            // convert buffer to little-endian bytes
            E::bytes_to_le(buf, self.bytes_per_sample);

            // update MD5 sum with little-endian bytes
            self.encoder.update_md5(buf);

            // encode fresh FLAC frame
            self.encoder
                .encode(self.frame.fill_from_buf::<LittleEndian>(buf))?;

            encoded_frames += 1;
        }
        // TODO - use truncate_front whenever that stabilizes
        self.buf.drain(0..self.frame_byte_size * encoded_frames);

        // indicate whole buffer's been consumed
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        // we don't want to flush a partial frame to disk,
        // but we can at least flush our internal writer
        self.encoder.writer.flush()
    }
}

impl<W: std::io::Write + std::io::Seek, E: crate::byteorder::Endianness> Drop for FlacWriter<W, E> {
    fn drop(&mut self) {
        let _ = self.finalize_inner();
    }
}

/// A FLAC writer which accepts samples as signed integers
///
/// # Example
///
/// ```
/// use flac_codec::{
///     encode::{FlacSampleWriter, Options},
///     decode::{FlacSampleReader, FlacSampleRead},
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,           // our wrapped writer
///     Options::default(),  // default encoding options
///     44100,               // sample rate
///     16,                  // bits-per-sample
///     1,                   // channel count
///     Some(1000),          // total samples
/// ).unwrap();
///
/// // write 1000 samples
/// let written_samples = (0..1000).collect::<Vec<i32>>();
/// assert!(writer.write(&written_samples).is_ok());
///
/// // finalize writing file
/// assert!(writer.finalize().is_ok());
///
/// flac.rewind().unwrap();
///
/// // open reader around written FLAC file
/// let mut reader = FlacSampleReader::new(flac).unwrap();
///
/// // read 1000 samples
/// let mut read_samples = vec![0; 1000];
/// assert!(matches!(reader.read(&mut read_samples), Ok(1000)));
///
/// // ensure they match
/// assert_eq!(read_samples, written_samples);
/// ```
pub struct FlacSampleWriter<W: std::io::Write + std::io::Seek> {
    // the wrapped encoder
    encoder: Encoder<W>,
    // samples that make up a partial FLAC frame
    // in channel-interleaved order
    // (must de-interleave later in case someone writes
    // only partial set of channels in a single write call)
    sample_buf: VecDeque<i32>,
    // a whole set of samples for a FLAC frame
    frame: Frame,
    // size of a single frame in samples
    frame_sample_size: usize,
    // size of a single PCM frame in samples
    pcm_frame_size: usize,
    // size of a single sample in bytes
    bytes_per_sample: usize,
    // size of single set of channel-independent samples in bytes
    // whether the encoder has finalized the file
    finalized: bool,
}

impl<W: std::io::Write + std::io::Seek> FlacSampleWriter<W> {
    /// Creates new FLAC writer with the given parameters
    ///
    /// `sample_rate` must be between 0 (for non-audio streams) and 2²⁰.
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// Note that if `total_samples` is indicated,
    /// the number of samples written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    /// Returns error if any of the encoding parameters are invalid.
    pub fn new(
        writer: W,
        options: Options,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u8,
        total_samples: Option<u64>,
    ) -> Result<Self, Error> {
        let bits_per_sample: SignedBitCount<32> = bits_per_sample
            .try_into()
            .map_err(|_| Error::InvalidBitsPerSample)?;

        let bytes_per_sample = u32::from(bits_per_sample).div_ceil(8) as usize;

        let pcm_frame_size = usize::from(channels);

        Ok(Self {
            sample_buf: VecDeque::default(),
            frame: Frame::empty(channels.into(), bits_per_sample.into()),
            bytes_per_sample,
            pcm_frame_size,
            frame_sample_size: pcm_frame_size * options.block_size as usize,
            encoder: Encoder::new(
                writer,
                options,
                sample_rate,
                bits_per_sample,
                channels,
                total_samples
                    .map(|samples| {
                        exact_div(samples, channels.into())
                            .ok_or(Error::SamplesNotDivisibleByChannels)
                            .and_then(|s| NonZero::new(s).ok_or(Error::InvalidTotalSamples))
                    })
                    .transpose()?,
            )?,
            finalized: false,
        })
    }

    /// Creates new FLAC writer with CDDA parameters
    ///
    /// Sample rate is 44100 Hz, bits-per-sample is 16,
    /// channels is 2.
    ///
    /// Note that if `total_samples` is indicated,
    /// the number of samples written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    /// Returns error if any of the encoding parameters are invalid.
    pub fn new_cdda(
        writer: W,
        options: Options,
        total_samples: Option<u64>,
    ) -> Result<Self, Error> {
        Self::new(writer, options, 44100, 16, 2, total_samples)
    }

    /// Given a set of samples, writes them to the FLAC file
    ///
    /// Samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// This may output 0 or more actual FLAC frames,
    /// depending on the quantity of samples and the amount
    /// previously written.
    pub fn write(&mut self, samples: &[i32]) -> Result<(), Error> {
        // dump whole set of samples into our internal buffer
        self.sample_buf.extend(samples);

        // encode as many FLAC frames as possible (which may be 0)
        let mut encoded_frames = 0;
        for buf in self
            .sample_buf
            .make_contiguous()
            .chunks_exact_mut(self.frame_sample_size)
        {
            // update running MD5 sum calculation
            // since samples are already interleaved in channel order
            update_md5(&mut self.encoder, buf, self.bytes_per_sample);

            // encode fresh FLAC frame
            self.encoder.encode(self.frame.fill_from_samples(buf))?;

            encoded_frames += 1;
        }
        self.sample_buf
            .drain(0..self.frame_sample_size * encoded_frames);

        Ok(())
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            self.finalized = true;

            // encode as many samples possible into final frame, if necessary
            if !self.sample_buf.is_empty() {
                // truncate buffer to whole PCM frames
                let buf = self.sample_buf.make_contiguous();
                let buf_len = buf.len();
                let buf = &mut buf[..(buf_len - buf_len % self.pcm_frame_size)];

                // update running MD5 sum calculation
                // since samples are already interleaved in channel order
                update_md5(&mut self.encoder, buf, self.bytes_per_sample);

                // encode final FLAC frame
                self.encoder.encode(self.frame.fill_from_samples(buf))?;
            }

            self.encoder.finalize_inner()
        } else {
            Ok(())
        }
    }

    /// Attempt to finalize stream
    ///
    /// It is necessary to finalize the FLAC encoder
    /// so that it will write any partially unwritten samples
    /// to the stream and update the [`crate::metadata::Streaminfo`] and [`crate::metadata::SeekTable`] blocks
    /// with their final values.
    ///
    /// Dropping the encoder will attempt to finalize the stream
    /// automatically, but will ignore any errors that may occur.
    pub fn finalize(mut self) -> Result<(), Error> {
        self.finalize_inner()?;
        Ok(())
    }
}

impl FlacSampleWriter<BufWriter<File>> {
    /// Creates new FLAC file at the given path
    ///
    /// `sample_rate` must be between 0 (for non-audio streams) and 2²⁰.
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of bytes written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    #[inline]
    pub fn create<P: AsRef<Path>>(
        path: P,
        options: Options,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u8,
        total_samples: Option<u64>,
    ) -> Result<Self, Error> {
        FlacSampleWriter::new(
            BufWriter::new(options.create(path)?),
            options,
            sample_rate,
            bits_per_sample,
            channels,
            total_samples,
        )
    }

    /// Creates new FLAC file at the given path with CDDA parameters
    ///
    /// Sample rate is 44100 Hz, bits-per-sample is 16,
    /// channels is 2.
    ///
    /// Note that if `total_bytes` is indicated,
    /// the number of bytes written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    #[inline]
    pub fn create_cdda<P: AsRef<Path>>(
        path: P,
        options: Options,
        total_samples: Option<u64>,
    ) -> Result<Self, Error> {
        Self::create(path, options, 44100, 16, 2, total_samples)
    }
}

/// A FLAC writer which operates on streamed output
///
/// Because this encodes FLAC frames without any metadata
/// blocks or finalizing, it does not need to be seekable.
///
/// # Example
///
/// ```
/// use flac_codec::{
///     decode::{FlacStreamReader, FrameBuf},
///     encode::{FlacStreamWriter, Options},
/// };
/// use std::io::{Cursor, Seek};
/// use bitstream_io::SignedBitCount;
///
/// let mut flac = Cursor::new(vec![]);
///
/// let samples = (0..100).collect::<Vec<i32>>();
///
/// let mut w = FlacStreamWriter::new(&mut flac, Options::default());
///
/// // write a single FLAC frame with some samples
/// w.write(
///     44100,  // sample rate
///     1,      // channels
///     16,     // bits-per-sample
///     &samples,
/// ).unwrap();
///
/// flac.rewind().unwrap();
///
/// let mut r = FlacStreamReader::new(&mut flac);
///
/// // read a single FLAC frame with some samples
/// assert_eq!(
///     r.read().unwrap(),
///     FrameBuf {
///         samples: &samples,
///         sample_rate: 44100,
///         channels: 1,
///         bits_per_sample: 16,
///     },
/// );
/// ```
pub struct FlacStreamWriter<W> {
    // the writer we're outputting to
    writer: W,
    // various encoding optins
    options: EncoderOptions,
    // various encoder caches
    caches: EncodingCaches,
    // a whole set of samples for a FLAC frame
    frame: Frame,
    // the current frame number
    frame_number: FrameNumber,
}

impl<W: std::io::Write> FlacStreamWriter<W> {
    /// Creates new stream writer
    pub fn new(writer: W, options: Options) -> Self {
        Self {
            writer,
            options: EncoderOptions {
                max_partition_order: options.max_partition_order,
                mid_side: options.mid_side,
                seektable_interval: options.seektable_interval,
                max_lpc_order: options.max_lpc_order,
                window: options.window,
                exhaustive_channel_correlation: options.exhaustive_channel_correlation,
            },
            caches: EncodingCaches::default(),
            frame: Frame::default(),
            frame_number: FrameNumber::default(),
        }
    }

    /// Writes a whole FLAC frame to our output stream
    ///
    /// Samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// This write a whole FLAC frame to the output stream on each call.
    ///
    /// # Errors
    ///
    /// Returns an error of any of the parameters are invalid
    /// or if an I/O error occurs when writing to the stream.
    pub fn write(
        &mut self,
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u32,
        samples: &[i32],
    ) -> Result<(), Error> {
        use crate::crc::{Crc16, CrcWriter};
        use crate::stream::{BitsPerSample, FrameHeader, SampleRate};

        let bits_per_sample: SignedBitCount<32> = bits_per_sample
            .try_into()
            .map_err(|_| Error::NonSubsetBitsPerSample)?;

        // samples must divide evenly into channels
        if samples.len() % usize::from(channels) != 0 {
            return Err(Error::SamplesNotDivisibleByChannels);
        } else if !(1..=8).contains(&channels) {
            return Err(Error::ExcessiveChannels);
        }

        self.frame
            .resize(bits_per_sample.into(), channels.into(), 0);
        self.frame.fill_from_samples(samples);

        // block size must be valid
        let block_size: crate::stream::BlockSize<u16> = crate::stream::BlockSize::try_from(
            u16::try_from(self.frame.pcm_frames()).map_err(|_| Error::InvalidBlockSize)?,
        )
        .map_err(|_| Error::InvalidBlockSize)?;

        // sample rate must be valid for subset streams
        let sample_rate: SampleRate<u32> = sample_rate.try_into().and_then(|rate| match rate {
            SampleRate::Streaminfo(_) => Err(Error::NonSubsetSampleRate),
            rate => Ok(rate),
        })?;

        // bits-per-sample must be valid for subset streams
        let header_bits_per_sample = match BitsPerSample::from(bits_per_sample) {
            BitsPerSample::Streaminfo(_) => return Err(Error::NonSubsetBitsPerSample),
            bps => bps,
        };

        let mut w: CrcWriter<_, Crc16> = CrcWriter::new(&mut self.writer);
        let mut bw: BitWriter<CrcWriter<&mut W, Crc16>, BigEndian>;

        match self
            .frame
            .channels()
            .collect::<ArrayVec<&[i32], MAX_CHANNELS>>()
            .as_slice()
        {
            [channel] => {
                FrameHeader {
                    blocking_strategy: false,
                    frame_number: self.frame_number,
                    block_size: (channel.len() as u16)
                        .try_into()
                        .expect("frame cannot be empty"),
                    sample_rate,
                    bits_per_sample: header_bits_per_sample,
                    channel_assignment: ChannelAssignment::Independent(Independent::Mono),
                }
                .write_subset(&mut w)?;

                bw = BitWriter::new(w);

                self.caches.channels.resize_with(1, ChannelCache::default);

                encode_subframe(
                    &self.options,
                    &mut self.caches.channels[0],
                    CorrelatedChannel::independent(bits_per_sample, channel),
                )?
                .playback(&mut bw)?;
            }
            [left, right] if self.options.exhaustive_channel_correlation => {
                let Correlated {
                    channel_assignment,
                    channels: [channel_0, channel_1],
                } = correlate_channels_exhaustive(
                    &self.options,
                    &mut self.caches.correlated,
                    [left, right],
                    bits_per_sample,
                )?;

                FrameHeader {
                    blocking_strategy: false,
                    frame_number: self.frame_number,
                    block_size,
                    sample_rate,
                    bits_per_sample: header_bits_per_sample,
                    channel_assignment,
                }
                .write_subset(&mut w)?;

                bw = BitWriter::new(w);

                channel_0.playback(&mut bw)?;
                channel_1.playback(&mut bw)?;
            }
            [left, right] => {
                let Correlated {
                    channel_assignment,
                    channels: [channel_0, channel_1],
                } = correlate_channels(
                    &self.options,
                    &mut self.caches.correlated,
                    [left, right],
                    bits_per_sample,
                );

                FrameHeader {
                    blocking_strategy: false,
                    frame_number: self.frame_number,
                    block_size,
                    sample_rate,
                    bits_per_sample: header_bits_per_sample,
                    channel_assignment,
                }
                .write_subset(&mut w)?;

                self.caches.channels.resize_with(2, ChannelCache::default);
                let [cache_0, cache_1] = self.caches.channels.get_disjoint_mut([0, 1]).unwrap();
                let (channel_0, channel_1) = join(
                    || encode_subframe(&self.options, cache_0, channel_0),
                    || encode_subframe(&self.options, cache_1, channel_1),
                );

                bw = BitWriter::new(w);

                channel_0?.playback(&mut bw)?;
                channel_1?.playback(&mut bw)?;
            }
            channels => {
                // non-stereo frames are always encoded independently
                FrameHeader {
                    blocking_strategy: false,
                    frame_number: self.frame_number,
                    block_size,
                    sample_rate,
                    bits_per_sample: header_bits_per_sample,
                    channel_assignment: ChannelAssignment::Independent(
                        channels.len().try_into().expect("invalid channel count"),
                    ),
                }
                .write_subset(&mut w)?;

                bw = BitWriter::new(w);

                self.caches
                    .channels
                    .resize_with(channels.len(), ChannelCache::default);

                vec_map(
                    self.caches.channels.iter_mut().zip(channels).collect(),
                    |(cache, channel)| {
                        encode_subframe(
                            &self.options,
                            cache,
                            CorrelatedChannel::independent(bits_per_sample, channel),
                        )
                    },
                )
                .into_iter()
                .try_for_each(|r| r.and_then(|r| r.playback(bw.by_ref()).map_err(Error::Io)))?;
            }
        }

        let crc16: u16 = bw.aligned_writer()?.checksum().into();
        bw.write_from(crc16)?;

        if self.frame_number.try_increment().is_err() {
            self.frame_number = FrameNumber::default();
        }

        Ok(())
    }

    /// Writes a whole FLAC frame to our output stream with CDDA parameters
    ///
    /// Samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// This write a whole FLAC frame to the output stream on each call.
    ///
    /// # Errors
    ///
    /// Returns an error of any of the parameters are invalid
    /// or if an I/O error occurs when writing to the stream.
    pub fn write_cdda(&mut self, samples: &[i32]) -> Result<(), Error> {
        self.write(44100, 2, 16, samples)
    }
}

fn update_md5<W: std::io::Write + std::io::Seek>(
    encoder: &mut Encoder<W>,
    samples: &[i32],
    bytes_per_sample: usize,
) {
    use crate::byteorder::{Endianness, LittleEndian};

    match bytes_per_sample {
        1 => {
            for s in samples {
                encoder.update_md5(&LittleEndian::i8_to_bytes(*s as i8));
            }
        }
        2 => {
            for s in samples {
                encoder.update_md5(&LittleEndian::i16_to_bytes(*s as i16));
            }
        }
        3 => {
            for s in samples {
                encoder.update_md5(&LittleEndian::i24_to_bytes(*s));
            }
        }
        4 => {
            for s in samples {
                encoder.update_md5(&LittleEndian::i32_to_bytes(*s));
            }
        }
        _ => panic!("unsupported number of bytes per sample"),
    }
}

/// The interval of seek points to generate
#[derive(Copy, Clone, Debug)]
pub enum SeekTableInterval {
    ///Generate seekpoint every nth seconds
    Seconds(NonZero<u8>),
    /// Generate seekpoint every nth frames
    Frames(NonZero<usize>),
}

impl Default for SeekTableInterval {
    fn default() -> Self {
        Self::Seconds(NonZero::new(10).unwrap())
    }
}

impl SeekTableInterval {
    // decimates full set of seekpoints based on the requested
    // seektable style, or returns None if no seektable is requested
    fn filter<'s>(
        self,
        sample_rate: u32,
        seekpoints: impl IntoIterator<Item = EncoderSeekPoint> + 's,
    ) -> Box<dyn Iterator<Item = EncoderSeekPoint> + 's> {
        match self {
            Self::Seconds(seconds) => {
                let nth_sample = u64::from(u32::from(seconds.get()) * sample_rate);
                let mut offset = 0;
                Box::new(seekpoints.into_iter().filter(move |point| {
                    if point.range().contains(&offset) {
                        offset += nth_sample;
                        true
                    } else {
                        false
                    }
                }))
            }
            Self::Frames(frames) => Box::new(seekpoints.into_iter().step_by(frames.get())),
        }
    }
}

/// FLAC encoding options
#[derive(Clone, Debug)]
pub struct Options {
    // whether to clobber existing file
    clobber: bool,
    block_size: u16,
    max_partition_order: u32,
    mid_side: bool,
    metadata: BlockList,
    seektable_interval: Option<SeekTableInterval>,
    max_lpc_order: Option<NonZero<u8>>,
    window: Window,
    exhaustive_channel_correlation: bool,
}

impl Default for Options {
    fn default() -> Self {
        // a dummy placeholder value
        // since we can't know the stream parameters yet
        let mut metadata = BlockList::new(Streaminfo {
            minimum_block_size: 0,
            maximum_block_size: 0,
            minimum_frame_size: None,
            maximum_frame_size: None,
            sample_rate: 0,
            channels: NonZero::new(1).unwrap(),
            bits_per_sample: SignedBitCount::new::<4>(),
            total_samples: None,
            md5: None,
        });

        metadata.insert(crate::metadata::Padding {
            size: 4096u16.into(),
        });

        Self {
            clobber: false,
            block_size: 4096,
            mid_side: true,
            max_partition_order: 5,
            metadata,
            seektable_interval: Some(SeekTableInterval::default()),
            max_lpc_order: NonZero::new(8),
            window: Window::default(),
            exhaustive_channel_correlation: true,
        }
    }
}

impl Options {
    /// Sets new block size
    ///
    /// Block size must be ≥ 16
    ///
    /// For subset streams, this must be ≤ 4608
    /// if the sample rate is ≤ 48 kHz -
    /// or ≤ 16384 for higher sample rates.
    pub fn block_size(self, block_size: u16) -> Result<Self, OptionsError> {
        match block_size {
            0..16 => Err(OptionsError::InvalidBlockSize),
            16.. => Ok(Self { block_size, ..self }),
        }
    }

    /// Sets new maximum LPC order
    ///
    /// The valid range is: 0 < `max_lpc_order` ≤ 32
    ///
    /// A value of `None` means that no LPC subframes will be encoded.
    pub fn max_lpc_order(self, max_lpc_order: Option<u8>) -> Result<Self, OptionsError> {
        Ok(Self {
            max_lpc_order: max_lpc_order
                .map(|o| {
                    o.try_into()
                        .ok()
                        .filter(|o| *o <= NonZero::new(32).unwrap())
                        .ok_or(OptionsError::InvalidLpcOrder)
                })
                .transpose()?,
            ..self
        })
    }

    /// Sets maximum residual partion order.
    ///
    /// The valid range is: 0 ≤ `max_partition_order` ≤ 15
    pub fn max_partition_order(self, max_partition_order: u32) -> Result<Self, OptionsError> {
        match max_partition_order {
            0..=15 => Ok(Self {
                max_partition_order,
                ..self
            }),
            16.. => Err(OptionsError::InvalidMaxPartitions),
        }
    }

    /// Whether to use mid-side encoding
    ///
    /// The default is `true`.
    pub fn mid_side(self, mid_side: bool) -> Self {
        Self { mid_side, ..self }
    }

    /// The windowing function to use for input samples
    pub fn window(self, window: Window) -> Self {
        Self { window, ..self }
    }

    /// Whether to calculate the best channel correlation quickly
    ///
    /// The default is `false`
    pub fn fast_channel_correlation(self, fast: bool) -> Self {
        Self {
            exhaustive_channel_correlation: !fast,
            ..self
        }
    }

    /// Updates size of padding block
    ///
    /// `size` must be < 2²⁴
    ///
    /// If `size` is set to 0, removes the block entirely.
    ///
    /// The default is to add a 4096 byte padding block.
    pub fn padding(mut self, size: u32) -> Result<Self, OptionsError> {
        use crate::metadata::Padding;

        match size
            .try_into()
            .map_err(|_| OptionsError::ExcessivePadding)?
        {
            BlockSize::ZERO => self.metadata.remove::<Padding>(),
            size => self.metadata.update::<Padding>(|p| {
                p.size = size;
            }),
        }
        Ok(self)
    }

    /// Remove any padding blocks from metadata
    ///
    /// This makes the file smaller, but will likely require
    /// rewriting it if any metadata needs to be modified later.
    pub fn no_padding(mut self) -> Self {
        self.metadata.remove::<crate::metadata::Padding>();
        self
    }

    /// Adds new tag to comment metadata block
    ///
    /// Creates new [`crate::metadata::VorbisComment`] block if not already present.
    pub fn tag<S>(mut self, field: &str, value: S) -> Self
    where
        S: std::fmt::Display,
    {
        self.metadata
            .update::<VorbisComment>(|vc| vc.insert(field, value));
        self
    }

    /// Replaces entire [`crate::metadata::VorbisComment`] metadata block
    ///
    /// This may be more convenient when adding many fields at once.
    pub fn comment(mut self, comment: VorbisComment) -> Self {
        self.metadata.insert(comment);
        self
    }

    /// Add new [`crate::metadata::Picture`] block to metadata
    ///
    /// Files may contain multiple [`crate::metadata::Picture`] blocks,
    /// and this adds a new block each time it is used.
    pub fn picture(mut self, picture: Picture) -> Self {
        self.metadata.insert(picture);
        self
    }

    /// Add new [`crate::metadata::Cuesheet`] block to metadata
    ///
    /// Files may (theoretically) contain multiple [`crate::metadata::Cuesheet`] blocks,
    /// and this adds a new block each time it is used.
    ///
    /// In practice, CD images almost always use only a single
    /// cue sheet.
    pub fn cuesheet(mut self, cuesheet: Cuesheet) -> Self {
        self.metadata.insert(cuesheet);
        self
    }

    /// Add new [`crate::metadata::Application`] block to metadata
    ///
    /// Files may contain multiple [`crate::metadata::Application`] blocks,
    /// and this adds a new block each time it is used.
    pub fn application(mut self, application: Application) -> Self {
        self.metadata.insert(application);
        self
    }

    /// Generate [`crate::metadata::SeekTable`] with the given number of seconds between seek points
    ///
    /// The default is to generate a SEEKTABLE with 10 seconds between seek points.
    ///
    /// If `seconds` is 0, removes the SEEKTABLE block.
    ///
    /// The interval between seek points may be larger than requested
    /// if the encoder's block size is larger than the seekpoint interval.
    pub fn seektable_seconds(mut self, seconds: u8) -> Self {
        // note that we can't drop a placeholder seektable
        // into the metadata blocks until we know
        // the sample rate and total samples of our stream
        self.seektable_interval = NonZero::new(seconds).map(SeekTableInterval::Seconds);
        self
    }

    /// Generate [`crate::metadata::SeekTable`] with the given number of FLAC frames between seek points
    ///
    /// If `frames` is 0, removes the SEEKTABLE block
    pub fn seektable_frames(mut self, frames: usize) -> Self {
        self.seektable_interval = NonZero::new(frames).map(SeekTableInterval::Frames);
        self
    }

    /// Do not generate a seektable in our encoded file
    pub fn no_seektable(self) -> Self {
        Self {
            seektable_interval: None,
            ..self
        }
    }

    /// Overwrites existing file if it already exists
    ///
    /// This only applies to encoding functions which
    /// create files from paths.
    ///
    /// The default is to not overwrite files
    /// if they already exist.
    pub fn overwrite(mut self) -> Self {
        self.clobber = true;
        self
    }

    /// Returns the fastest encoding options
    ///
    /// These are tuned to encode as quickly as possible.
    pub fn fast() -> Self {
        Self {
            block_size: 1152,
            mid_side: false,
            max_partition_order: 3,
            max_lpc_order: None,
            exhaustive_channel_correlation: false,
            ..Self::default()
        }
    }

    /// Returns the fastest encoding options
    ///
    /// These are tuned to encode files as small as possible.
    pub fn best() -> Self {
        Self {
            block_size: 4096,
            mid_side: true,
            max_partition_order: 6,
            max_lpc_order: NonZero::new(12),
            ..Self::default()
        }
    }

    /// Creates files according to whether clobber is set or not
    fn create<P: AsRef<Path>>(&self, path: P) -> std::io::Result<File> {
        if self.clobber {
            File::create(path)
        } else {
            use std::fs::OpenOptions;

            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path.as_ref())
        }
    }
}

/// An error when specifying encoding options
#[derive(Debug)]
pub enum OptionsError {
    /// Selected block size is too small
    InvalidBlockSize,
    /// Maximum LPC order is too large
    InvalidLpcOrder,
    /// Maximum residual partitions is too large
    InvalidMaxPartitions,
    /// Selected padding size is too large
    ExcessivePadding,
}

impl std::error::Error for OptionsError {}

impl std::fmt::Display for OptionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidBlockSize => "block size must be >= 16".fmt(f),
            Self::InvalidLpcOrder => "maximum LPC order must be <= 32".fmt(f),
            Self::InvalidMaxPartitions => "max partition order must be <= 15".fmt(f),
            Self::ExcessivePadding => "padding size is too large for block".fmt(f),
        }
    }
}

/// A cut-down version of Options without the metadata blocks
struct EncoderOptions {
    max_partition_order: u32,
    mid_side: bool,
    seektable_interval: Option<SeekTableInterval>,
    max_lpc_order: Option<NonZero<u8>>,
    window: Window,
    exhaustive_channel_correlation: bool,
}

/// The method to use for windowing the input signal
#[derive(Copy, Clone, Debug)]
pub enum Window {
    /// Basic rectangular window
    Rectangle,
    /// Hann window
    Hann,
    /// Tukey window
    Tukey(f32),
}

// TODO - add more windowing options

impl Window {
    fn generate(&self, window: &mut [f64]) {
        use std::f64::consts::PI;

        match self {
            Self::Rectangle => window.fill(1.0),
            Self::Hann => {
                // verified output against reference implementation

                let np =
                    f64::from(u16::try_from(window.len()).expect("window size too large")) - 1.0;

                window.iter_mut().zip(0u16..).for_each(|(w, n)| {
                    *w = 0.5 - 0.5 * (2.0 * PI * f64::from(n) / np).cos();
                });
            }
            Self::Tukey(p) => match p {
                // verified output against reference implementation
                ..=0.0 => {
                    window.fill(1.0);
                }
                1.0.. => {
                    Self::Hann.generate(window);
                }
                0.0..1.0 => {
                    match ((f64::from(*p) / 2.0 * window.len() as f64) as usize).checked_sub(1) {
                        Some(np) => match window.get_disjoint_mut([
                            0..np,
                            np..window.len() - np,
                            window.len() - np..window.len(),
                        ]) {
                            Ok([first, mid, last]) => {
                                // u16 is maximum block size
                                let np = u16::try_from(np).expect("window size too large");

                                for ((x, y), n) in
                                    first.iter_mut().zip(last.iter_mut().rev()).zip(0u16..)
                                {
                                    *x = 0.5 - 0.5 * (PI * f64::from(n) / f64::from(np)).cos();
                                    *y = *x;
                                }
                                mid.fill(1.0);
                            }
                            Err(_) => {
                                window.fill(1.0);
                            }
                        },
                        None => {
                            window.fill(1.0);
                        }
                    }
                }
                _ => {
                    Self::Tukey(0.5).generate(window);
                }
            },
        }
    }

    fn apply<'w>(
        &self,
        window: &mut Vec<f64>,
        cache: &'w mut Vec<f64>,
        samples: &[i32],
    ) -> &'w [f64] {
        if window.len() != samples.len() {
            // need to re-generate window to fit samples
            window.resize(samples.len(), 0.0);
            self.generate(window);
        }

        // window signal into cache and return cached slice
        cache.clear();
        cache.extend(samples.iter().zip(window).map(|(s, w)| f64::from(*s) * *w));
        cache.as_slice()
    }
}

impl Default for Window {
    fn default() -> Self {
        Self::Tukey(0.5)
    }
}

#[derive(Default)]
struct EncodingCaches {
    channels: Vec<ChannelCache>,
    correlated: CorrelationCache,
}

#[derive(Default)]
struct CorrelationCache {
    // the average channel samples
    average_samples: Vec<i32>,
    // the difference channel samples
    difference_samples: Vec<i32>,

    left_cache: ChannelCache,
    right_cache: ChannelCache,
    average_cache: ChannelCache,
    difference_cache: ChannelCache,
}

#[derive(Default)]
struct ChannelCache {
    fixed: FixedCache,
    fixed_output: BitRecorder<u32, BigEndian>,
    lpc: LpcCache,
    lpc_output: BitRecorder<u32, BigEndian>,
    constant_output: BitRecorder<u32, BigEndian>,
    verbatim_output: BitRecorder<u32, BigEndian>,
    wasted: Vec<i32>,
}

#[derive(Default)]
struct FixedCache {
    // FIXED subframe buffers, one per order 1-4
    fixed_buffers: [Vec<i32>; 4],
}

#[derive(Default)]
struct LpcCache {
    window: Vec<f64>,
    windowed: Vec<f64>,
    residuals: Vec<i32>,
}

/// A FLAC encoder
struct Encoder<W: std::io::Write + std::io::Seek> {
    // the writer we're outputting to
    writer: Counter<W>,
    // the stream's starting offset in the writer, in bytes
    start: u64,
    // various encoding options
    options: EncoderOptions,
    // various encoder caches
    caches: EncodingCaches,
    // our metadata blocks
    blocks: BlockList,
    // our stream's sample rate
    sample_rate: SampleRate<u32>,
    // the current frame number
    frame_number: FrameNumber,
    // the number of channel-independent samples written
    samples_written: u64,
    // all seekpoints
    seekpoints: Vec<EncoderSeekPoint>,
    // our running MD5 calculation
    md5: md5::Context,
    // whether the encoder has finalized the file
    finalized: bool,
}

impl<W: std::io::Write + std::io::Seek> Encoder<W> {
    const MAX_SAMPLES: u64 = 68_719_476_736;

    fn new(
        mut writer: W,
        options: Options,
        sample_rate: u32,
        bits_per_sample: impl TryInto<SignedBitCount<32>>,
        channels: u8,
        total_samples: Option<NonZero<u64>>,
    ) -> Result<Self, Error> {
        let mut blocks = options.metadata;

        *blocks.streaminfo_mut() = Streaminfo {
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
            channels: (1..=8)
                .contains(&channels)
                .then_some(channels)
                .and_then(NonZero::new)
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

        // insert a dummy SeekTable to be populated later
        if let Some(total_samples) = total_samples {
            if let Some(placeholders) = options.seektable_interval.map(|s| {
                s.filter(
                    sample_rate,
                    EncoderSeekPoint::placeholders(total_samples.get(), options.block_size),
                )
            }) {
                use crate::metadata::SeekTable;

                blocks.insert(SeekTable {
                    points: placeholders
                        .take(SeekTable::MAX_POINTS)
                        .map(|p| p.into())
                        .collect(),
                });
            }
        }

        let start = writer.stream_position()?;
        write_blocks(writer.by_ref(), blocks.blocks())?;

        Ok(Self {
            start,
            writer: Counter::new(writer),
            options: EncoderOptions {
                max_partition_order: options.max_partition_order,
                mid_side: options.mid_side,
                seektable_interval: options.seektable_interval,
                max_lpc_order: options.max_lpc_order,
                window: options.window,
                exhaustive_channel_correlation: options.exhaustive_channel_correlation,
            },
            caches: EncodingCaches::default(),
            sample_rate: blocks
                .streaminfo()
                .sample_rate
                .try_into()
                .expect("invalid sample rate"),
            blocks,
            frame_number: FrameNumber::default(),
            samples_written: 0,
            seekpoints: Vec::new(),
            md5: md5::Context::new(),
            finalized: false,
        })
    }

    /// Updates running MD5 calculation with signed, little-endian bytes
    fn update_md5(&mut self, frame_bytes: &[u8]) {
        self.md5.consume(frame_bytes)
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
    fn encode(&mut self, frame: &Frame) -> Result<(), Error> {
        // drop in a new seekpoint
        self.seekpoints.push(EncoderSeekPoint {
            sample_offset: self.samples_written,
            byte_offset: Some(self.writer.count),
            frame_samples: frame.pcm_frames() as u16,
        });

        // update running total of samples written
        self.samples_written += frame.pcm_frames() as u64;
        if let Some(total_samples) = self.blocks.streaminfo().total_samples {
            if self.samples_written > total_samples.get() {
                return Err(Error::ExcessiveTotalSamples);
            }
        }

        encode_frame(
            &self.options,
            &mut self.caches,
            &mut self.writer,
            self.blocks.streaminfo_mut(),
            &mut self.frame_number,
            self.sample_rate,
            frame.channels().collect(),
        )
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            use crate::metadata::SeekTable;

            self.finalized = true;

            // update SEEKTABLE metadata block with final values
            if let Some(encoded_points) = self
                .options
                .seektable_interval
                .map(|s| s.filter(self.sample_rate.into(), self.seekpoints.iter().cloned()))
            {
                match self.blocks.get_pair_mut() {
                    (Some(SeekTable { points }), _) => {
                        // placeholder SEEKTABLE already in place,
                        // so no need to adjust PADDING to fit

                        points
                            .iter_mut()
                            .zip(encoded_points)
                            .for_each(|(o, i)| *o = i.into());
                    }
                    (None, Some(crate::metadata::Padding { size: padding_size })) => {
                        // no SEEKTABLE, but there is a PADDING block,
                        // so try to shrink PADDING to fit SEEKTABLE

                        use crate::metadata::MetadataBlock;

                        let seektable = SeekTable {
                            points: encoded_points.map(|p| p.into()).collect(),
                        };
                        if let Some(new_padding_size) = seektable
                            .total_size()
                            .and_then(|seektable_size| padding_size.checked_sub(seektable_size))
                        {
                            *padding_size = new_padding_size;
                            self.blocks.insert(seektable);
                        }
                    }
                    (None, None) => { /* no seektable or padding, so nothing to do */ }
                }
            }

            // verify or update final sample count
            match &mut self.blocks.streaminfo_mut().total_samples {
                Some(expected) => {
                    // ensure final sample count matches
                    if expected.get() != self.samples_written {
                        return Err(Error::SampleCountMismatch);
                    }
                }
                expected @ None => {
                    // update final sample count if possible
                    if self.samples_written < Self::MAX_SAMPLES {
                        *expected =
                            Some(NonZero::new(self.samples_written).ok_or(Error::NoSamples)?);
                    } else {
                        // TODO - should I just leave this blank
                        // if too many samples are written?
                        return Err(Error::ExcessiveTotalSamples);
                    }
                }
            }

            // update STREAMINFO MD5 sum
            self.blocks.streaminfo_mut().md5 = Some(self.md5.clone().compute().0);

            // rewrite metadata blocks, relative to the beginning
            // of the stream
            let writer = self.writer.stream();
            writer.seek(std::io::SeekFrom::Start(self.start))?;
            write_blocks(writer.by_ref(), self.blocks.blocks())
        } else {
            Ok(())
        }
    }
}

impl<W: std::io::Write + std::io::Seek> Drop for Encoder<W> {
    fn drop(&mut self) {
        let _ = self.finalize_inner();
    }
}

// Unlike regular SeekPoints, which can have placeholder variants,
// these are always defined to be something.  A byte offset
// of None indicates a dummy encoder point
#[derive(Debug, Clone)]
struct EncoderSeekPoint {
    sample_offset: u64,
    byte_offset: Option<u64>,
    frame_samples: u16,
}

impl EncoderSeekPoint {
    // generates set of placeholder points
    fn placeholders(total_samples: u64, block_size: u16) -> impl Iterator<Item = EncoderSeekPoint> {
        (0..total_samples)
            .step_by(usize::from(block_size))
            .map(move |sample_offset| EncoderSeekPoint {
                sample_offset,
                byte_offset: None,
                frame_samples: u16::try_from(total_samples - sample_offset)
                    .map(|s| s.min(block_size))
                    .unwrap_or(block_size),
            })
    }

    // returns sample range of point
    fn range(&self) -> std::ops::Range<u64> {
        self.sample_offset..(self.sample_offset + u64::from(self.frame_samples))
    }
}

impl From<EncoderSeekPoint> for SeekPoint {
    fn from(p: EncoderSeekPoint) -> Self {
        match p.byte_offset {
            Some(byte_offset) => Self::Defined {
                sample_offset: p.sample_offset,
                byte_offset,
                frame_samples: p.frame_samples,
            },
            None => Self::Placeholder,
        }
    }
}

/// Given a FLAC stream, generates new seek table
///
/// Though encoders should add seek tables by default,
/// sometimes one isn't present.  This function takes
/// an existing FLAC file stream and generates a new
/// seek table suitable for adding to the file's metadata
/// via the [`crate::metadata::update`] function.
///
/// The stream should be rewound to the beginning of the file.
///
/// # Errors
///
/// Returns any error from the underlying stream.
///
/// # Example
/// ```
/// use flac_codec::{
///     encode::{FlacSampleWriter, Options, SeekTableInterval, generate_seektable},
///     metadata::{SeekTable, read_block},
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// // add a seekpoint every second
/// let options = Options::default().seektable_seconds(1);
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,         // our wrapped writer
///     options,           // our seektable options
///     44100,             // sample rate
///     16,                // bits-per-sample
///     1,                 // channel count
///     Some(60 * 44100),  // one minute's worth of samples
/// ).unwrap();
///
/// // write one minute's worth of samples
/// writer.write(vec![0; 60 * 44100].as_slice()).unwrap();
///
/// // finalize writing file
/// assert!(writer.finalize().is_ok());
///
/// flac.rewind().unwrap();
///
/// // get existing seektable
/// let original_seektable = match read_block::<_, SeekTable>(&mut flac) {
///     Ok(Some(seektable)) => seektable,
///     _ => panic!("seektable not found"),
/// };
///
/// flac.rewind().unwrap();
///
/// // generate new seektable, also with seekpoints every second
/// let new_seektable = generate_seektable(
///     flac,
///     SeekTableInterval::Seconds(1.try_into().unwrap())
/// ).unwrap();
///
/// // ensure both seektables are identical
/// assert_eq!(original_seektable, new_seektable);
/// ```
pub fn generate_seektable<R: std::io::Read>(
    r: R,
    interval: SeekTableInterval,
) -> Result<crate::metadata::SeekTable, Error> {
    use crate::{
        metadata::{Metadata, SeekTable},
        stream::FrameIterator,
    };

    let iter = FrameIterator::new(r)?;
    let metadata_len = iter.metadata_len();
    let sample_rate = iter.sample_rate();
    let mut sample_offset = 0;

    iter.map(|r| {
        r.map(|(frame, offset)| EncoderSeekPoint {
            sample_offset,
            byte_offset: Some(offset - metadata_len),
            frame_samples: frame.header.block_size.into(),
        })
        .inspect(|p| {
            sample_offset += u64::from(p.frame_samples);
        })
    })
    .collect::<Result<Vec<_>, _>>()
    .map(|seekpoints| SeekTable {
        points: interval
            .filter(sample_rate, seekpoints)
            .take(SeekTable::MAX_POINTS)
            .map(|p| p.into())
            .collect(),
    })
}

fn encode_frame<W>(
    options: &EncoderOptions,
    cache: &mut EncodingCaches,
    mut writer: W,
    streaminfo: &mut Streaminfo,
    frame_number: &mut FrameNumber,
    sample_rate: SampleRate<u32>,
    frame: ArrayVec<&[i32], MAX_CHANNELS>,
) -> Result<(), Error>
where
    W: std::io::Write,
{
    use crate::Counter;
    use crate::crc::{Crc16, CrcWriter};
    use crate::stream::FrameHeader;
    use bitstream_io::BigEndian;

    debug_assert!(!frame.is_empty());

    let size = Counter::new(writer.by_ref());
    let mut w: CrcWriter<_, Crc16> = CrcWriter::new(size);
    let mut bw: BitWriter<CrcWriter<Counter<&mut W>, Crc16>, BigEndian>;

    match frame.as_slice() {
        [channel] => {
            FrameHeader {
                blocking_strategy: false,
                frame_number: *frame_number,
                block_size: (channel.len() as u16)
                    .try_into()
                    .expect("frame cannot be empty"),
                sample_rate,
                bits_per_sample: streaminfo.bits_per_sample.into(),
                channel_assignment: ChannelAssignment::Independent(Independent::Mono),
            }
            .write(&mut w, streaminfo)?;

            bw = BitWriter::new(w);

            cache.channels.resize_with(1, ChannelCache::default);

            encode_subframe(
                options,
                &mut cache.channels[0],
                CorrelatedChannel::independent(streaminfo.bits_per_sample, channel),
            )?
            .playback(&mut bw)?;
        }
        [left, right] if options.exhaustive_channel_correlation => {
            let Correlated {
                channel_assignment,
                channels: [channel_0, channel_1],
            } = correlate_channels_exhaustive(
                options,
                &mut cache.correlated,
                [left, right],
                streaminfo.bits_per_sample,
            )?;

            FrameHeader {
                blocking_strategy: false,
                frame_number: *frame_number,
                block_size: (frame[0].len() as u16)
                    .try_into()
                    .expect("frame cannot be empty"),
                sample_rate,
                bits_per_sample: streaminfo.bits_per_sample.into(),
                channel_assignment,
            }
            .write(&mut w, streaminfo)?;

            bw = BitWriter::new(w);

            channel_0.playback(&mut bw)?;
            channel_1.playback(&mut bw)?;
        }
        [left, right] => {
            let Correlated {
                channel_assignment,
                channels: [channel_0, channel_1],
            } = correlate_channels(
                options,
                &mut cache.correlated,
                [left, right],
                streaminfo.bits_per_sample,
            );

            FrameHeader {
                blocking_strategy: false,
                frame_number: *frame_number,
                block_size: (frame[0].len() as u16)
                    .try_into()
                    .expect("frame cannot be empty"),
                sample_rate,
                bits_per_sample: streaminfo.bits_per_sample.into(),
                channel_assignment,
            }
            .write(&mut w, streaminfo)?;

            cache.channels.resize_with(2, ChannelCache::default);
            let [cache_0, cache_1] = cache.channels.get_disjoint_mut([0, 1]).unwrap();
            let (channel_0, channel_1) = join(
                || encode_subframe(options, cache_0, channel_0),
                || encode_subframe(options, cache_1, channel_1),
            );

            bw = BitWriter::new(w);

            channel_0?.playback(&mut bw)?;
            channel_1?.playback(&mut bw)?;
        }
        channels => {
            // non-stereo frames are always encoded independently

            FrameHeader {
                blocking_strategy: false,
                frame_number: *frame_number,
                block_size: (channels[0].len() as u16)
                    .try_into()
                    .expect("frame cannot be empty"),
                sample_rate,
                bits_per_sample: streaminfo.bits_per_sample.into(),
                channel_assignment: ChannelAssignment::Independent(
                    frame.len().try_into().expect("invalid channel count"),
                ),
            }
            .write(&mut w, streaminfo)?;

            bw = BitWriter::new(w);

            cache
                .channels
                .resize_with(channels.len(), ChannelCache::default);

            vec_map(
                cache.channels.iter_mut().zip(channels).collect(),
                |(cache, channel)| {
                    encode_subframe(
                        options,
                        cache,
                        CorrelatedChannel::independent(streaminfo.bits_per_sample, channel),
                    )
                },
            )
            .into_iter()
            .try_for_each(|r| r.and_then(|r| r.playback(bw.by_ref()).map_err(Error::Io)))?;
        }
    }

    let crc16: u16 = bw.aligned_writer()?.checksum().into();
    bw.write_from(crc16)?;

    frame_number.try_increment()?;

    // update minimum and maximum frame size values
    if let s @ Some(size) = u32::try_from(bw.into_writer().into_writer().count)
        .ok()
        .filter(|size| *size < Streaminfo::MAX_FRAME_SIZE)
        .and_then(NonZero::new)
    {
        match &mut streaminfo.minimum_frame_size {
            Some(min_size) => {
                *min_size = size.min(*min_size);
            }
            min_size @ None => {
                *min_size = s;
            }
        }

        match &mut streaminfo.maximum_frame_size {
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

struct Correlated<C> {
    channel_assignment: ChannelAssignment,
    channels: [C; 2],
}

struct CorrelatedChannel<'c> {
    samples: &'c [i32],
    bits_per_sample: SignedBitCount<32>,
    // whether all samples are known to be 0
    all_0: bool,
}

impl<'c> CorrelatedChannel<'c> {
    fn independent(bits_per_sample: SignedBitCount<32>, samples: &'c [i32]) -> Self {
        Self {
            all_0: samples.iter().all(|s| *s == 0),
            bits_per_sample,
            samples,
        }
    }
}

fn correlate_channels<'c>(
    options: &EncoderOptions,
    CorrelationCache {
        average_samples,
        difference_samples,
        ..
    }: &'c mut CorrelationCache,
    [left, right]: [&'c [i32]; 2],
    bits_per_sample: SignedBitCount<32>,
) -> Correlated<CorrelatedChannel<'c>> {
    match bits_per_sample.checked_add::<32>(1) {
        Some(difference_bits_per_sample) if options.mid_side => {
            let mut left_abs_sum = 0;
            let mut right_abs_sum = 0;
            let mut mid_abs_sum = 0;
            let mut side_abs_sum = 0;

            join(
                || {
                    average_samples.clear();
                    average_samples.extend(
                        left.iter()
                            .inspect(|s| left_abs_sum += u64::from(s.unsigned_abs()))
                            .zip(
                                right
                                    .iter()
                                    .inspect(|s| right_abs_sum += u64::from(s.unsigned_abs())),
                            )
                            .map(|(l, r)| (l + r) >> 1)
                            .inspect(|s| mid_abs_sum += u64::from(s.unsigned_abs())),
                    );
                },
                || {
                    difference_samples.clear();
                    difference_samples.extend(
                        left.iter()
                            .zip(right)
                            .map(|(l, r)| l - r)
                            .inspect(|s| side_abs_sum += u64::from(s.unsigned_abs())),
                    );
                },
            );

            match [
                (
                    ChannelAssignment::Independent(Independent::Stereo),
                    left_abs_sum + right_abs_sum,
                ),
                (ChannelAssignment::LeftSide, left_abs_sum + side_abs_sum),
                (ChannelAssignment::SideRight, side_abs_sum + right_abs_sum),
                (ChannelAssignment::MidSide, mid_abs_sum + side_abs_sum),
            ]
            .into_iter()
            .min_by_key(|(_, total)| *total)
            .unwrap()
            .0
            {
                channel_assignment @ ChannelAssignment::LeftSide => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: left,
                            bits_per_sample,
                            all_0: left_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: difference_samples,
                            bits_per_sample: difference_bits_per_sample,
                            all_0: side_abs_sum == 0,
                        },
                    ],
                },
                channel_assignment @ ChannelAssignment::SideRight => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: difference_samples,
                            bits_per_sample: difference_bits_per_sample,
                            all_0: side_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: right,
                            bits_per_sample,
                            all_0: right_abs_sum == 0,
                        },
                    ],
                },
                channel_assignment @ ChannelAssignment::MidSide => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: average_samples,
                            bits_per_sample,
                            all_0: mid_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: difference_samples,
                            bits_per_sample: difference_bits_per_sample,
                            all_0: side_abs_sum == 0,
                        },
                    ],
                },
                channel_assignment @ ChannelAssignment::Independent(_) => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: left,
                            bits_per_sample,
                            all_0: left_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: right,
                            bits_per_sample,
                            all_0: right_abs_sum == 0,
                        },
                    ],
                },
            }
        }
        Some(difference_bits_per_sample) => {
            let mut left_abs_sum = 0;
            let mut right_abs_sum = 0;
            let mut side_abs_sum = 0;

            difference_samples.clear();
            difference_samples.extend(
                left.iter()
                    .inspect(|s| left_abs_sum += u64::from(s.unsigned_abs()))
                    .zip(
                        right
                            .iter()
                            .inspect(|s| right_abs_sum += u64::from(s.unsigned_abs())),
                    )
                    .map(|(l, r)| l - r)
                    .inspect(|s| side_abs_sum += u64::from(s.unsigned_abs())),
            );

            match [
                (ChannelAssignment::LeftSide, left_abs_sum + side_abs_sum),
                (ChannelAssignment::SideRight, side_abs_sum + right_abs_sum),
                (
                    ChannelAssignment::Independent(Independent::Stereo),
                    left_abs_sum + right_abs_sum,
                ),
            ]
            .into_iter()
            .min_by_key(|(_, total)| *total)
            .unwrap()
            .0
            {
                channel_assignment @ ChannelAssignment::LeftSide => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: left,
                            bits_per_sample,
                            all_0: left_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: difference_samples,
                            bits_per_sample: difference_bits_per_sample,
                            all_0: side_abs_sum == 0,
                        },
                    ],
                },
                channel_assignment @ ChannelAssignment::SideRight => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: difference_samples,
                            bits_per_sample: difference_bits_per_sample,
                            all_0: side_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: right,
                            bits_per_sample,
                            all_0: right_abs_sum == 0,
                        },
                    ],
                },
                ChannelAssignment::MidSide => unreachable!(),
                channel_assignment @ ChannelAssignment::Independent(_) => Correlated {
                    channel_assignment,
                    channels: [
                        CorrelatedChannel {
                            samples: left,
                            bits_per_sample,
                            all_0: left_abs_sum == 0,
                        },
                        CorrelatedChannel {
                            samples: right,
                            bits_per_sample,
                            all_0: right_abs_sum == 0,
                        },
                    ],
                },
            }
        }
        None => {
            // 32 bps stream, so forego difference channel
            // and encode them both indepedently

            Correlated {
                channel_assignment: ChannelAssignment::Independent(Independent::Stereo),
                channels: [
                    CorrelatedChannel::independent(bits_per_sample, left),
                    CorrelatedChannel::independent(bits_per_sample, right),
                ],
            }
        }
    }
}

fn correlate_channels_exhaustive<'c>(
    options: &EncoderOptions,
    CorrelationCache {
        average_samples,
        difference_samples,
        left_cache,
        right_cache,
        average_cache,
        difference_cache,
        ..
    }: &'c mut CorrelationCache,
    [left, right]: [&'c [i32]; 2],
    bits_per_sample: SignedBitCount<32>,
) -> Result<Correlated<&'c BitRecorder<u32, BigEndian>>, Error> {
    let (left_recorder, right_recorder) = try_join(
        || {
            encode_subframe(
                options,
                left_cache,
                CorrelatedChannel {
                    samples: left,
                    bits_per_sample,
                    all_0: false,
                },
            )
        },
        || {
            encode_subframe(
                options,
                right_cache,
                CorrelatedChannel {
                    samples: right,
                    bits_per_sample,
                    all_0: false,
                },
            )
        },
    )?;

    match bits_per_sample.checked_add::<32>(1) {
        Some(difference_bits_per_sample) if options.mid_side => {
            let (average_recorder, difference_recorder) = try_join(
                || {
                    average_samples.clear();
                    average_samples
                        .extend(left.iter().zip(right.iter()).map(|(l, r)| (l + r) >> 1));
                    encode_subframe(
                        options,
                        average_cache,
                        CorrelatedChannel {
                            samples: average_samples,
                            bits_per_sample,
                            all_0: false,
                        },
                    )
                },
                || {
                    difference_samples.clear();
                    difference_samples.extend(left.iter().zip(right).map(|(l, r)| l - r));
                    encode_subframe(
                        options,
                        difference_cache,
                        CorrelatedChannel {
                            samples: difference_samples,
                            bits_per_sample: difference_bits_per_sample,
                            all_0: false,
                        },
                    )
                },
            )?;

            match [
                (
                    ChannelAssignment::Independent(Independent::Stereo),
                    left_recorder.written() + right_recorder.written(),
                ),
                (
                    ChannelAssignment::LeftSide,
                    left_recorder.written() + difference_recorder.written(),
                ),
                (
                    ChannelAssignment::SideRight,
                    difference_recorder.written() + right_recorder.written(),
                ),
                (
                    ChannelAssignment::MidSide,
                    average_recorder.written() + difference_recorder.written(),
                ),
            ]
            .into_iter()
            .min_by_key(|(_, total)| *total)
            .unwrap()
            .0
            {
                channel_assignment @ ChannelAssignment::LeftSide => Ok(Correlated {
                    channel_assignment,
                    channels: [left_recorder, difference_recorder],
                }),
                channel_assignment @ ChannelAssignment::SideRight => Ok(Correlated {
                    channel_assignment,
                    channels: [difference_recorder, right_recorder],
                }),
                channel_assignment @ ChannelAssignment::MidSide => Ok(Correlated {
                    channel_assignment,
                    channels: [average_recorder, difference_recorder],
                }),
                channel_assignment @ ChannelAssignment::Independent(_) => Ok(Correlated {
                    channel_assignment,
                    channels: [left_recorder, right_recorder],
                }),
            }
        }
        Some(difference_bits_per_sample) => {
            let difference_recorder = {
                difference_samples.clear();
                difference_samples.extend(left.iter().zip(right).map(|(l, r)| l - r));
                encode_subframe(
                    options,
                    difference_cache,
                    CorrelatedChannel {
                        samples: difference_samples,
                        bits_per_sample: difference_bits_per_sample,
                        all_0: false,
                    },
                )?
            };

            match [
                (
                    ChannelAssignment::Independent(Independent::Stereo),
                    left_recorder.written() + right_recorder.written(),
                ),
                (
                    ChannelAssignment::LeftSide,
                    left_recorder.written() + difference_recorder.written(),
                ),
                (
                    ChannelAssignment::SideRight,
                    difference_recorder.written() + right_recorder.written(),
                ),
            ]
            .into_iter()
            .min_by_key(|(_, total)| *total)
            .unwrap()
            .0
            {
                channel_assignment @ ChannelAssignment::LeftSide => Ok(Correlated {
                    channel_assignment,
                    channels: [left_recorder, difference_recorder],
                }),
                channel_assignment @ ChannelAssignment::SideRight => Ok(Correlated {
                    channel_assignment,
                    channels: [difference_recorder, right_recorder],
                }),
                ChannelAssignment::MidSide => unreachable!(),
                channel_assignment @ ChannelAssignment::Independent(_) => Ok(Correlated {
                    channel_assignment,
                    channels: [left_recorder, right_recorder],
                }),
            }
        }
        None => {
            // 32 bps stream, so forego difference channel
            // and encode them both indepedently

            Ok(Correlated {
                channel_assignment: ChannelAssignment::Independent(Independent::Stereo),
                channels: [left_recorder, right_recorder],
            })
        }
    }
}

fn encode_subframe<'c>(
    options: &EncoderOptions,
    ChannelCache {
        fixed: fixed_cache,
        fixed_output,
        lpc: lpc_cache,
        lpc_output,
        constant_output,
        verbatim_output,
        wasted,
    }: &'c mut ChannelCache,
    CorrelatedChannel {
        samples: channel,
        bits_per_sample,
        all_0,
    }: CorrelatedChannel,
) -> Result<&'c BitRecorder<u32, BigEndian>, Error> {
    const WASTED_MAX: NonZero<u32> = NonZero::new(32).unwrap();

    debug_assert!(!channel.is_empty());

    if all_0 {
        // all samples are 0
        constant_output.clear();
        encode_constant_subframe(constant_output, channel[0], bits_per_sample, 0)?;
        return Ok(constant_output);
    }

    // determine any wasted bits
    let (channel, bits_per_sample, wasted_bps) =
        match channel.iter().try_fold(WASTED_MAX, |acc, sample| {
            NonZero::new(sample.trailing_zeros()).map(|sample| sample.min(acc))
        }) {
            None => (channel, bits_per_sample, 0),
            Some(WASTED_MAX) => {
                constant_output.clear();
                encode_constant_subframe(constant_output, channel[0], bits_per_sample, 0)?;
                return Ok(constant_output);
            }
            Some(wasted_bps) => {
                let wasted_bps = wasted_bps.get();
                wasted.clear();
                wasted.extend(channel.iter().map(|sample| sample >> wasted_bps));
                (
                    wasted.as_slice(),
                    bits_per_sample.checked_sub(wasted_bps).unwrap(),
                    wasted_bps,
                )
            }
        };

    fixed_output.clear();

    let best = match options.max_lpc_order {
        Some(max_lpc_order) if channel.len() > usize::from(max_lpc_order.get()) => {
            lpc_output.clear();

            match join(
                || {
                    encode_fixed_subframe(
                        options,
                        fixed_cache,
                        fixed_output,
                        channel,
                        bits_per_sample,
                        wasted_bps,
                    )
                },
                || {
                    encode_lpc_subframe(
                        options,
                        max_lpc_order,
                        lpc_cache,
                        lpc_output,
                        channel,
                        bits_per_sample,
                        wasted_bps,
                    )
                },
            ) {
                (Ok(()), Ok(())) => [fixed_output, lpc_output]
                    .into_iter()
                    .min_by_key(|c| c.written())
                    .unwrap(),
                (Err(_), Ok(())) => lpc_output,
                (Ok(()), Err(_)) => fixed_output,
                (Err(_), Err(_)) => {
                    verbatim_output.clear();
                    encode_verbatim_subframe(
                        verbatim_output,
                        channel,
                        bits_per_sample,
                        wasted_bps,
                    )?;
                    return Ok(verbatim_output);
                }
            }
        }
        _ => {
            match encode_fixed_subframe(
                options,
                fixed_cache,
                fixed_output,
                channel,
                bits_per_sample,
                wasted_bps,
            ) {
                Ok(()) => fixed_output,
                Err(_) => {
                    verbatim_output.clear();
                    encode_verbatim_subframe(
                        verbatim_output,
                        channel,
                        bits_per_sample,
                        wasted_bps,
                    )?;
                    return Ok(verbatim_output);
                }
            }
        }
    };

    let verbatim_len = channel.len() as u32 * u32::from(bits_per_sample);

    if best.written() < verbatim_len {
        Ok(best)
    } else {
        verbatim_output.clear();
        encode_verbatim_subframe(verbatim_output, channel, bits_per_sample, wasted_bps)?;
        Ok(verbatim_output)
    }
}

fn encode_constant_subframe<W: BitWrite>(
    writer: &mut W,
    sample: i32,
    bits_per_sample: SignedBitCount<32>,
    wasted_bps: u32,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    writer.build(&SubframeHeader {
        type_: SubframeHeaderType::Constant,
        wasted_bps,
    })?;

    writer
        .write_signed_counted(bits_per_sample, sample)
        .map_err(Error::Io)
}

fn encode_verbatim_subframe<W: BitWrite>(
    writer: &mut W,
    channel: &[i32],
    bits_per_sample: SignedBitCount<32>,
    wasted_bps: u32,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    writer.build(&SubframeHeader {
        type_: SubframeHeaderType::Verbatim,
        wasted_bps,
    })?;

    channel
        .iter()
        .try_for_each(|i| writer.write_signed_counted(bits_per_sample, *i))?;

    Ok(())
}

fn encode_fixed_subframe<W: BitWrite>(
    options: &EncoderOptions,
    FixedCache {
        fixed_buffers: buffers,
    }: &mut FixedCache,
    writer: &mut W,
    channel: &[i32],
    bits_per_sample: SignedBitCount<32>,
    wasted_bps: u32,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    // calculate residuals for FIXED subframe orders 0-4
    // (or fewer, if we don't have enough samples)
    let (order, warm_up, residuals) = {
        let mut fixed_orders = ArrayVec::<&[i32], 5>::new();
        fixed_orders.push(channel);

        // accumulate a set of FIXED diffs
        'outer: for buf in buffers.iter_mut() {
            let prev_order = fixed_orders.last().unwrap();
            match prev_order.split_at_checked(1) {
                Some((_, r)) => {
                    buf.clear();
                    for (n, p) in r.iter().zip(*prev_order) {
                        match n.checked_sub(*p) {
                            Some(v) => {
                                buf.push(v);
                            }
                            None => break 'outer,
                        }
                    }
                    if buf.is_empty() {
                        break;
                    } else {
                        fixed_orders.push(buf.as_slice());
                    }
                }
                None => break,
            }
        }

        let min_fixed = fixed_orders.last().unwrap().len();

        // choose diff with the smallest abs sum
        fixed_orders
            .into_iter()
            .enumerate()
            .min_by_key(|(_, residuals)| {
                residuals[(residuals.len() - min_fixed)..]
                    .iter()
                    .map(|r| u64::from(r.unsigned_abs()))
                    .sum::<u64>()
            })
            .map(|(order, residuals)| (order as u8, &channel[0..order], residuals))
            .unwrap()
    };

    writer.build(&SubframeHeader {
        type_: SubframeHeaderType::Fixed { order },
        wasted_bps,
    })?;

    warm_up
        .iter()
        .try_for_each(|sample: &i32| writer.write_signed_counted(bits_per_sample, *sample))?;

    write_residuals(options, writer, order.into(), residuals)
}

fn encode_lpc_subframe<W: BitWrite>(
    options: &EncoderOptions,
    max_lpc_order: NonZero<u8>,
    cache: &mut LpcCache,
    writer: &mut W,
    channel: &[i32],
    bits_per_sample: SignedBitCount<32>,
    wasted_bps: u32,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    let LpcSubframeParameters {
        warm_up,
        residuals,
        parameters:
            LpcParameters {
                order,
                precision,
                shift,
                coefficients,
            },
    } = LpcSubframeParameters::best(options, bits_per_sample, max_lpc_order, cache, channel)?;

    writer.build(&SubframeHeader {
        type_: SubframeHeaderType::Lpc { order },
        wasted_bps,
    })?;

    for sample in warm_up {
        writer.write_signed_counted(bits_per_sample, *sample)?;
    }

    writer.write_count::<0b1111>(
        precision
            .count()
            .checked_sub(1)
            .ok_or(Error::InvalidQlpPrecision)?,
    )?;

    writer.write::<5, i32>(shift as i32)?;

    for coeff in coefficients {
        writer.write_signed_counted(precision, coeff)?;
    }

    write_residuals(options, writer, order.get().into(), residuals)
}

struct LpcSubframeParameters<'w, 'r> {
    parameters: LpcParameters,
    warm_up: &'w [i32],
    residuals: &'r [i32],
}

impl<'w, 'r> LpcSubframeParameters<'w, 'r> {
    fn best(
        options: &EncoderOptions,
        bits_per_sample: SignedBitCount<32>,
        max_lpc_order: NonZero<u8>,
        LpcCache {
            residuals,
            window,
            windowed,
        }: &'r mut LpcCache,
        channel: &'w [i32],
    ) -> Result<Self, ResidualOverflow> {
        let parameters = LpcParameters::best(
            options,
            bits_per_sample,
            max_lpc_order,
            window,
            windowed,
            channel,
        );

        Self::encode_residuals(&parameters, channel, residuals).map(|(warm_up, residuals)| Self {
            warm_up,
            residuals,
            parameters,
        })
    }

    fn encode_residuals(
        parameters: &LpcParameters,
        channel: &'w [i32],
        residuals: &'r mut Vec<i32>,
    ) -> Result<(&'w [i32], &'r [i32]), ResidualOverflow> {
        residuals.clear();

        for split in usize::from(parameters.order.get())..channel.len() {
            let (previous, current) = channel.split_at(split);

            residuals.push(
                current[0]
                    .checked_sub(
                        (previous
                            .iter()
                            .rev()
                            .zip(&parameters.coefficients)
                            .map(|(x, y)| *x as i64 * *y as i64)
                            .sum::<i64>()
                            >> parameters.shift) as i32,
                    )
                    .ok_or(ResidualOverflow)?,
            );
        }

        Ok((
            &channel[0..parameters.order.get().into()],
            residuals.as_slice(),
        ))
    }
}

#[derive(Debug)]
struct ResidualOverflow;

impl From<ResidualOverflow> for Error {
    #[inline]
    fn from(_: ResidualOverflow) -> Self {
        Error::ResidualOverflow
    }
}

#[test]
fn test_residual_encoding_1() {
    let samples = [
        0, 16, 31, 44, 54, 61, 64, 63, 58, 49, 38, 24, 8, -8, -24, -38, -49, -58, -63, -64, -61,
        -54, -44, -31, -16,
    ];

    let expected_residuals = [
        2, 2, 2, 3, 3, 3, 2, 2, 3, 0, 0, 0, -1, -1, -1, -3, -2, -2, -2, -1, -1, 0, 0,
    ];

    let mut actual_residuals = Vec::with_capacity(expected_residuals.len());

    let (warm_up, residuals) = LpcSubframeParameters::encode_residuals(
        &LpcParameters {
            order: NonZero::new(2).unwrap(),
            precision: SignedBitCount::new::<7>(),
            shift: 5,
            coefficients: vec![59, -30],
        },
        &samples,
        &mut actual_residuals,
    )
    .unwrap();

    assert_eq!(warm_up, &samples[0..2]);
    assert_eq!(residuals, &expected_residuals);
}

#[test]
fn test_residual_encoding_2() {
    let samples = [
        64, 62, 56, 47, 34, 20, 4, -12, -27, -41, -52, -60, -63, -63, -60, -52, -41, -27, -12, 4,
        20, 34, 47, 56, 62,
    ];

    let expected_residuals = [
        2, 2, 0, 1, -1, -1, -1, -2, -2, -2, -1, -3, -2, 0, -1, 1, 0, 2, 2, 2, 4, 2, 4,
    ];

    let mut actual_residuals = Vec::with_capacity(expected_residuals.len());

    let (warm_up, residuals) = LpcSubframeParameters::encode_residuals(
        &LpcParameters {
            order: NonZero::new(2).unwrap(),
            precision: SignedBitCount::new::<7>(),
            shift: 5,
            coefficients: vec![58, -29],
        },
        &samples,
        &mut actual_residuals,
    )
    .unwrap();

    assert_eq!(warm_up, &samples[0..2]);
    assert_eq!(residuals, &expected_residuals);
}

#[derive(Debug)]
struct LpcParameters {
    order: NonZero<u8>,
    precision: SignedBitCount<15>,
    shift: u32,
    coefficients: Vec<i32>,
}

// There isn't any particular *best* way to determine
// the ideal LPC subframe parameters (though there are
// some worst ways, like choosing them at random).
// Even the reference implementation has changed its
// defaults over time.  So long as the subframe's residuals
// are calculated correctly, decoders don't care one way or another.
//
// I'll try to use an approach similar to the reference implementation's.

impl LpcParameters {
    fn best(
        options: &EncoderOptions,
        bits_per_sample: SignedBitCount<32>,
        max_lpc_order: NonZero<u8>,
        window: &mut Vec<f64>,
        windowed: &mut Vec<f64>,
        channel: &[i32],
    ) -> Self {
        debug_assert!(channel.len() > usize::from(max_lpc_order.get()));

        let precision = match channel.len() {
            0 => panic!("at least one sample required in channel"),
            1..=192 => SignedBitCount::new::<7>(),
            193..=384 => SignedBitCount::new::<8>(),
            385..=576 => SignedBitCount::new::<9>(),
            577..=1152 => SignedBitCount::new::<10>(),
            1153..=2304 => SignedBitCount::new::<11>(),
            2305..=4608 => SignedBitCount::new::<12>(),
            4609.. => SignedBitCount::new::<13>(),
        };

        let (order, lp_coeffs) = estimate_best_order(
            bits_per_sample,
            precision,
            channel
                .len()
                .try_into()
                .expect("excessive samples for subframe"),
            lp_coefficients(autocorrelate(
                options.window.apply(window, windowed, channel),
                max_lpc_order,
            )),
        );

        Self::quantize(order, lp_coeffs, precision)
    }

    fn quantize(order: NonZero<u8>, coeffs: Vec<f64>, precision: SignedBitCount<15>) -> Self {
        // verified output against reference implementation

        debug_assert!(coeffs.len() == usize::from(order.get()));

        let max_coeff = (1 << (u32::from(precision) - 1)) - 1;
        let min_coeff = -(1 << (u32::from(precision) - 1));

        let l = coeffs
            .iter()
            .map(|c| c.abs())
            .max_by(|x, y| x.total_cmp(y))
            .unwrap();

        if l > 0.0 {
            let shift: u32 = ((u32::from(precision) - 1) as i32 - ((l.log2().floor()) as i32) - 1)
                .clamp(0, (1 << 4) - 1)
                .try_into()
                .unwrap();

            let mut error = 0.0;

            Self {
                order,
                precision,
                shift,
                coefficients: coeffs
                    .into_iter()
                    .map(|lp_coeff| {
                        let sum: f64 = lp_coeff.mul_add((1 << shift) as f64, error);
                        let qlp_coeff = (sum.round() as i32).clamp(min_coeff, max_coeff);
                        error = sum - (qlp_coeff as f64);
                        qlp_coeff
                    })
                    .collect(),
            }
        } else {
            Self {
                order,
                precision,
                shift: 0,
                coefficients: vec![0; usize::from(order.get())],
            }
        }
    }
}

fn autocorrelate(windowed: &[f64], max_lpc_order: NonZero<u8>) -> Vec<f64> {
    // verified output against reference implementation

    let mut tail = windowed;
    let mut autocorrelated = Vec::with_capacity(max_lpc_order.get().into());

    for _ in 0..=max_lpc_order.get() {
        if tail.is_empty() {
            return autocorrelated;
        } else {
            autocorrelated.push(windowed.iter().zip(tail).map(|(x, y)| x * y).sum());
            tail.split_off_first();
        }
    }

    autocorrelated
}

#[test]
fn test_autocorrelation() {
    assert_eq!(autocorrelate(&[1.0], NonZero::new(1).unwrap()), &[1.0],);

    assert_eq!(
        autocorrelate(&[1.0, 2.0, 3.0, 4.0, 5.0], NonZero::new(4).unwrap()),
        &[55.0, 40.0, 26.0, 14.0, 5.0],
    );

    assert_eq!(
        autocorrelate(
            &[
                0.0, 16.0, 31.0, 44.0, 54.0, 61.0, 64.0, 63.0, 58.0, 49.0, 38.0, 24.0, 8.0, -8.0,
                -24.0, -38.0, -49.0, -58.0, -63.0, -64.0, -61.0, -54.0, -44.0, -31.0, -16.0,
            ],
            NonZero::new(4).unwrap()
        ),
        &[51408.0, 49792.0, 45304.0, 38466.0, 29914.0],
    )
}

#[derive(Debug)]
struct LpCoeff {
    coeffs: Vec<f64>,
    error: f64,
}

// returns a Vec of (coefficients, error) pairs
fn lp_coefficients(autocorrelated: Vec<f64>) -> Vec<LpCoeff> {
    // verified output against reference implementation

    match autocorrelated.len() {
        0 | 1 => panic!("must have at least 2 autocorrelation values"),
        _ => {
            let k = autocorrelated[1] / autocorrelated[0];
            let mut lp_coefficients = vec![LpCoeff {
                coeffs: vec![k],
                error: autocorrelated[0] * (1.0 - k.powi(2)),
            }];

            for i in 1..(autocorrelated.len() - 1) {
                if let [prev @ .., next] = &autocorrelated[0..=i + 1] {
                    let LpCoeff { coeffs, error } = lp_coefficients.last().unwrap();

                    let q = next
                        - prev
                            .iter()
                            .rev()
                            .zip(coeffs)
                            .map(|(x, y)| x * y)
                            .sum::<f64>();

                    let k = q / error;

                    lp_coefficients.push(LpCoeff {
                        coeffs: coeffs
                            .iter()
                            .zip(coeffs.iter().rev().map(|c| k * c))
                            .map(|(c1, c2)| (c1 - c2))
                            .chain(std::iter::once(k))
                            .collect(),
                        error: error * (1.0 - k.powi(2)),
                    });
                }
            }

            lp_coefficients
        }
    }
}

// returns (order, coeffs) pair
fn estimate_best_order(
    bits_per_sample: SignedBitCount<32>,
    precision: SignedBitCount<15>,
    sample_count: u16,
    coeffs: Vec<LpCoeff>,
) -> (NonZero<u8>, Vec<f64>) {
    // verified output against reference implementation

    debug_assert!(sample_count > 0);

    let error_scale = 0.5 / f64::from(sample_count);

    coeffs
        .into_iter()
        .take_while(|coeffs| coeffs.error > 0.0)
        .zip(1..)
        .map(|(LpCoeff { coeffs, error }, order)| {
            let header_bits =
                u32::from(order) * (u32::from(bits_per_sample) + u32::from(precision));
            let bits_per_residual =
                (error * error_scale).ln() / (2.0 * std::f64::consts::LN_2).max(0.0);
            let subframe_bits = bits_per_residual.mul_add(
                f64::from(sample_count - u16::from(order)),
                f64::from(header_bits),
            );
            (subframe_bits, order, coeffs)
        })
        .min_by(|(x, _, _), (y, _, _)| x.total_cmp(y))
        .and_then(|(_, order, coeffs)| Some((NonZero::new(order)?, coeffs)))
        .expect("coefficient list cannot be empty")
}

fn write_residuals<W: BitWrite>(
    options: &EncoderOptions,
    writer: &mut W,
    predictor_order: usize,
    residuals: &[i32],
) -> Result<(), Error> {
    use crate::stream::ResidualPartitionHeader;
    use bitstream_io::BitCount;

    const MAX_PARTITIONS: usize = 64;

    #[derive(Debug)]
    struct Partition<'r, const RICE_MAX: u32> {
        header: ResidualPartitionHeader<RICE_MAX>,
        residuals: &'r [i32],
    }

    impl<'r, const RICE_MAX: u32> Partition<'r, RICE_MAX> {
        fn new(partition: &'r [i32], estimated_bits: &mut u32) -> Option<Self> {
            let partition_samples = partition.len() as u16;
            if partition_samples == 0 {
                return None;
            }

            let partition_sum = partition
                .iter()
                .map(|i| u64::from(i.unsigned_abs()))
                .sum::<u64>();

            if partition_sum > 0 {
                let rice = if partition_sum > partition_samples.into() {
                    let bits_needed = ((partition_sum as f64) / f64::from(partition_samples))
                        .log2()
                        .ceil() as u32;

                    match BitCount::try_from(bits_needed).ok().filter(|rice| {
                        u32::from(*rice) < u32::from(BitCount::<RICE_MAX>::new::<RICE_MAX>())
                    }) {
                        Some(rice) => rice,
                        None => {
                            let escape_size = (partition
                                .iter()
                                .map(|i| u64::from(i.unsigned_abs()))
                                .sum::<u64>()
                                .ilog2()
                                + 2)
                            .try_into()
                            .ok()?;

                            *estimated_bits +=
                                u32::from(escape_size) * u32::from(partition_samples);

                            return Some(Self {
                                header: ResidualPartitionHeader::Escaped { escape_size },
                                residuals: partition,
                            });
                        }
                    }
                } else {
                    BitCount::new::<0>()
                };

                let partition_size: u32 = 4u32
                    + ((1 + u32::from(rice)) * u32::from(partition_samples))
                    + if u32::from(rice) > 0 {
                        u32::try_from(partition_sum >> (u32::from(rice) - 1)).ok()?
                    } else {
                        u32::try_from(partition_sum << 1).ok()?
                    }
                    - (u32::from(partition_samples) / 2);

                *estimated_bits += partition_size;

                Some(Partition {
                    header: ResidualPartitionHeader::Standard { rice },
                    residuals: partition,
                })
            } else {
                // all partition residuals are 0, so use a constant
                Some(Partition {
                    header: ResidualPartitionHeader::Constant,
                    residuals: partition,
                })
            }
        }
    }

    fn best_partitions<'r, const RICE_MAX: u32>(
        options: &EncoderOptions,
        block_size: usize,
        residuals: &'r [i32],
    ) -> ArrayVec<Partition<'r, RICE_MAX>, MAX_PARTITIONS> {
        (0..=block_size.trailing_zeros().min(options.max_partition_order))
            .map(|partition_order| 1 << partition_order)
            .take_while(|partition_count: &usize| partition_count.is_power_of_two())
            .filter_map(|partition_count| {
                let mut estimated_bits = 0;

                let partitions = residuals
                    .rchunks(block_size / partition_count)
                    .rev()
                    .map(|partition| Partition::new(partition, &mut estimated_bits))
                    .collect::<Option<ArrayVec<_, MAX_PARTITIONS>>>()
                    .filter(|p| !p.is_empty() && p.len().is_power_of_two())?;

                Some((partitions, estimated_bits))
            })
            .min_by_key(|(_, estimated_bits)| *estimated_bits)
            .map(|(partitions, _)| partitions)
            .unwrap_or_else(|| {
                std::iter::once(Partition {
                    header: ResidualPartitionHeader::Escaped {
                        escape_size: SignedBitCount::new::<0b11111>(),
                    },
                    residuals,
                })
                .collect()
            })
    }

    fn write_block<const RICE_MAX: u32, W: BitWrite>(
        options: &EncoderOptions,
        writer: &mut W,
        predictor_order: usize,
        residuals: &[i32],
    ) -> Result<(), Error> {
        let block_size = predictor_order + residuals.len();

        let partitions = best_partitions::<RICE_MAX>(options, block_size, residuals);

        writer.write::<4, u32>(partitions.len().ilog2())?; // partition order

        for Partition { header, residuals } in partitions {
            writer.build(&header)?;
            match header {
                ResidualPartitionHeader::Standard { rice } => {
                    let mask = rice.mask_lsb();

                    residuals.iter().try_for_each(|s| {
                        let (msb, lsb) = mask(if s.is_negative() {
                            ((-*s as u32 - 1) << 1) + 1
                        } else {
                            (*s as u32) << 1
                        });
                        writer.write_unary::<1>(msb)?;
                        writer.write_checked(lsb)
                    })?;
                }
                ResidualPartitionHeader::Escaped { escape_size } => {
                    residuals
                        .iter()
                        .try_for_each(|s| writer.write_signed_counted(escape_size, *s))?;
                }
                ResidualPartitionHeader::Constant => { /* nothing left to do */ }
            }
        }
        Ok(())
    }

    // TODO - we only support a coding method of 0
    writer.write::<2, u8>(0)?; // coding method
    write_block::<0b1111, W>(options, writer, predictor_order, residuals)
}

fn try_join<A, B, RA, RB, E>(oper_a: A, oper_b: B) -> Result<(RA, RB), E>
where
    A: FnOnce() -> Result<RA, E> + Send,
    B: FnOnce() -> Result<RB, E> + Send,
    RA: Send,
    RB: Send,
    E: Send,
{
    let (a, b) = join(oper_a, oper_b);
    Ok((a?, b?))
}

#[cfg(feature = "rayon")]
use rayon::join;

#[cfg(not(feature = "rayon"))]
fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    (oper_a(), oper_b())
}

#[cfg(feature = "rayon")]
fn vec_map<T, U, F>(src: Vec<T>, f: F) -> Vec<U>
where
    T: Send,
    U: Send,
    F: Fn(T) -> U + Send + Sync,
{
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    src.into_par_iter().map(f).collect()
}

#[cfg(not(feature = "rayon"))]
fn vec_map<T, U, F>(src: Vec<T>, f: F) -> Vec<U>
where
    T: Send,
    U: Send,
    F: Fn(T) -> U,
{
    src.into_iter().map(f).collect()
}

fn exact_div<N>(n: N, rhs: N) -> Option<N>
where
    N: std::ops::Div<Output = N> + std::ops::Rem<Output = N> + std::cmp::PartialEq + Copy + Default,
{
    (n % rhs == N::default()).then_some(n / rhs)
}
