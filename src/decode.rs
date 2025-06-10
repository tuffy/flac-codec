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
use crate::metadata::{BlockList, BlockRef, SeekTable};
use bitstream_io::{BitRead, SignedBitCount};
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::num::NonZero;
use std::path::Path;

/// A FLAC reader which outputs PCM samples as bytes
///
/// # Example
///
/// ```
/// use flac_codec::{
///     byteorder::LittleEndian,
///     encode::{FlacWriter, EncodingOptions},
///     decode::FlacReader
/// };
/// use std::io::{Cursor, Read, Seek, Write};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// let mut writer = FlacWriter::endian(
///     &mut flac,                   // our wrapped writer
///     LittleEndian,                // .wav-style byte order
///     EncodingOptions::default(),  // default encoding options
///     44100,                       // sample rate
///     16,                          // bits-per-sample
///     1,                           // channel count
///     Some(2000),                  // total bytes
/// ).unwrap();
///
/// // write 1000 samples as signed, little-endian bytes
/// let written_bytes = (0..1000).map(i16::to_le_bytes).flatten().collect::<Vec<u8>>();
/// assert!(writer.write_all(&written_bytes).is_ok());
///
/// // finalize writing file
/// assert!(writer.finalize().is_ok());
///
/// flac.rewind().unwrap();
///
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
pub struct FlacReader<R, E> {
    // the wrapped decoder
    decoder: Decoder<R>,
    // decoded byte buffer
    buf: VecDeque<u8>,
    // the endianness of the bytes in our byte buffer
    endianness: std::marker::PhantomData<E>,
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> FlacReader<R, E> {
    /// Opens new FLAC reader which wraps the given reader
    ///
    /// The reader must be positioned at the start of the
    /// first FLAC frame.
    #[inline]
    pub fn new(reader: R) -> Result<Self, Error> {
        Ok(Self {
            decoder: Decoder::new(reader)?,
            buf: VecDeque::default(),
            endianness: std::marker::PhantomData,
        })
    }

    /// Opens new FLAC reader in the given endianness
    ///
    /// The reader must be positioned at the start of the
    /// first FLAC frame.
    #[inline]
    pub fn endian(reader: R, _endian: E) -> Result<Self, Error> {
        Self::new(reader)
    }

    /// Returns channel count
    ///
    /// From 1 to 8
    #[inline]
    pub fn channel_count(&self) -> NonZero<u8> {
        self.decoder.channel_count()
    }

    /// Returns sample rate, in Hz
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.decoder.sample_rate()
    }

    /// Returns decoder's bits-per-sample
    ///
    /// From 1 to 32
    #[inline]
    pub fn bits_per_sample(&self) -> u32 {
        self.decoder.bits_per_sample()
    }

    /// Returns total number of channel-independent samples, if known
    #[inline]
    pub fn total_samples(&self) -> Option<NonZero<u64>> {
        self.decoder.total_samples()
    }

    /// Returns MD5 of entire stream, if known
    ///
    /// MD5 is always calculated in terms of little-endian,
    /// signed, byte-aligned values.
    #[inline]
    pub fn md5(&self) -> Option<&[u8; 16]> {
        self.decoder.md5()
    }

    /// Returns iterator over all metadata blocks
    #[inline]
    pub fn metadata(&self) -> impl Iterator<Item = BlockRef<'_>> {
        self.decoder.metadata()
    }
}

impl<E: crate::byteorder::Endianness> FlacReader<BufReader<File>, E> {
    /// Opens FLAC file from the given path
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P, _endianness: E) -> Result<Self, Error> {
        FlacReader::new(BufReader::new(File::open(path.as_ref())?))
    }
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> std::io::Read for FlacReader<R, E> {
    /// Reads samples to the given buffer as bytes in our stream's endianness
    ///
    /// Returned samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// # Errors
    ///
    /// Returns any error that occurs when reading the stream,
    /// converted to an I/O error.
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.buf.is_empty() {
            match self.decoder.read_frame()? {
                Some(frame) => {
                    self.buf.resize(frame.bytes_len(), 0);
                    frame.to_buf::<E>(self.buf.make_contiguous());
                    self.buf.read(buf)
                }
                None => Ok(0),
            }
        } else {
            self.buf.read(buf)
        }
    }
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> std::io::BufRead for FlacReader<R, E> {
    /// Reads samples to the given buffer as bytes in our stream's endianness
    ///
    /// Returned samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// # Errors
    ///
    /// Returns any error that occurs when reading the stream,
    /// converted to an I/O error.
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.buf.is_empty() {
            match self.decoder.read_frame()? {
                Some(frame) => {
                    self.buf.resize(frame.bytes_len(), 0);
                    frame.to_buf::<E>(self.buf.make_contiguous());
                    self.buf.fill_buf()
                }
                None => Ok(&[]),
            }
        } else {
            self.buf.fill_buf()
        }
    }

    fn consume(&mut self, amt: usize) {
        self.buf.consume(amt)
    }
}

impl<R: std::io::Read + std::io::Seek, E: crate::byteorder::Endianness> std::io::Seek
    for FlacReader<R, E>
{
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        use std::cmp::Ordering;

        let streaminfo = self.decoder.blocks.streaminfo();

        let bytes_per_pcm_frame: u64 = (u32::from(streaminfo.bits_per_sample).div_ceil(8)
            * u32::from(streaminfo.channels.get()))
        .into();

        // the desired absolute position in the stream, in bytes
        let desired_pos: u64 =
            match pos {
                std::io::SeekFrom::Start(pos) => pos,
                std::io::SeekFrom::Current(pos) => {
                    // current position in bytes is current position in samples
                    // converted to bytes *minus* the un-consumed space in the buffer
                    // since the sample position is running ahead of the byte position
                    let original_pos: u64 = (self.decoder.current_sample * bytes_per_pcm_frame)
                        - (self.buf.len() as u64);

                    match pos.cmp(&0) {
                        Ordering::Less => original_pos.checked_sub(pos.unsigned_abs()).ok_or(
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "cannot seek below byte 0",
                            ),
                        )?,
                        Ordering::Equal => return Ok(original_pos),
                        Ordering::Greater => original_pos.checked_add(pos.unsigned_abs()).ok_or(
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "seek offset too large",
                            ),
                        )?,
                    }
                }
                std::io::SeekFrom::End(pos) => {
                    // if the total samples is unknown in streaminfo,
                    // we have no way to know where the file's end is
                    // (this is a very unusual case)
                    let max_pos: u64 = self.decoder.total_samples().map(|s| s.get()).ok_or(
                        std::io::Error::new(
                            std::io::ErrorKind::NotSeekable,
                            "total samples not known",
                        ),
                    )?;

                    match pos.cmp(&0) {
                        Ordering::Less => {
                            max_pos
                                .checked_sub(pos.unsigned_abs())
                                .ok_or(std::io::Error::new(
                                    std::io::ErrorKind::InvalidInput,
                                    "cannot seek below byte 0",
                                ))?
                        }
                        Ordering::Equal => max_pos,
                        Ordering::Greater => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "cannot seek beyond end of file",
                            ));
                        }
                    }
                }
            };

        // perform seek in stream to the desired sample
        // (this will usually be some position prior to the desired sample)
        let mut new_pos =
            self.decoder.seek(desired_pos / bytes_per_pcm_frame)? * bytes_per_pcm_frame;

        // seeking invalidates current buffer
        self.buf.clear();

        // skip bytes to reach desired sample
        while new_pos < desired_pos {
            use std::io::BufRead;

            let buf = self.fill_buf()?;

            if !buf.is_empty() {
                let to_skip = (usize::try_from(desired_pos - new_pos).unwrap()).min(buf.len());
                self.consume(to_skip);
                new_pos += to_skip as u64;
            } else {
                // attempting to seek beyond the end of the FLAC file
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "stream exhausted before sample reached",
                ));
            }
        }

        Ok(desired_pos)
    }
}

/// A FLAC reader which outputs PCM samples as signed integers
///
/// # Example
///
/// ```
/// use flac_codec::{encode::{FlacSampleWriter, EncodingOptions}, decode::FlacSampleReader};
/// use std::io::{Cursor, Seek};
/// use std::num::NonZero;
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,                   // our wrapped writer
///     EncodingOptions::default(),  // default encoding options
///     44100,                       // sample rate
///     16,                          // bits-per-sample
///     1,                           // channel count
///     NonZero::new(1000),          // total samples
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
/// let mut reader = FlacSampleReader::new(flac).unwrap();
///
/// // read 1000 samples
/// let mut read_samples = vec![0; 1000];
/// assert!(matches!(reader.read(&mut read_samples), Ok(1000)));
///
/// // ensure they match
/// assert_eq!(read_samples, written_samples);
/// ```
pub struct FlacSampleReader<R> {
    // the wrapped decoder
    decoder: Decoder<R>,
    // decoded sample buffer
    buf: VecDeque<i32>,
}

impl<R: std::io::Read> FlacSampleReader<R> {
    /// Opens new FLAC reader which wraps the given reader
    #[inline]
    pub fn new(reader: R) -> Result<Self, Error> {
        Ok(Self {
            decoder: Decoder::new(reader)?,
            buf: VecDeque::default(),
        })
    }

    /// Returns channel count
    ///
    /// From 1 to 8
    #[inline]
    pub fn channel_count(&self) -> NonZero<u8> {
        self.decoder.channel_count()
    }

    /// Returns sample rate, in Hz
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.decoder.sample_rate()
    }

    /// Returns decoder's bits-per-sample
    ///
    /// From 1 to 32
    #[inline]
    pub fn bits_per_sample(&self) -> u32 {
        self.decoder.bits_per_sample()
    }

    /// Returns total number of channel-independent samples, if known
    #[inline]
    pub fn total_samples(&self) -> Option<NonZero<u64>> {
        self.decoder.total_samples()
    }

    /// Returns MD5 of entire stream, if known
    ///
    /// MD5 is always calculated in terms of little-endian,
    /// signed, byte-aligned values.
    #[inline]
    pub fn md5(&self) -> Option<&[u8; 16]> {
        self.decoder.md5()
    }

    /// Returns iterator over all metadata blocks
    #[inline]
    pub fn metadata(&self) -> impl Iterator<Item = BlockRef<'_>> {
        self.decoder.metadata()
    }

    /// Attempts to fill the buffer with samples and returns quantity read
    ///
    /// Returned samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// # Errors
    ///
    /// Returns error if some error occurs reading FLAC file
    pub fn read(&mut self, samples: &mut [i32]) -> Result<usize, Error> {
        if self.buf.is_empty() {
            match self.decoder.read_frame()? {
                Some(frame) => {
                    self.buf.extend(frame.iter());
                }
                None => return Ok(0),
            }
        }

        let to_consume = samples.len().min(self.buf.len());
        for (i, o) in samples.iter_mut().zip(self.buf.drain(0..to_consume)) {
            *i = o;
        }
        Ok(to_consume)
    }

    /// Returns complete buffer of all read samples
    ///
    /// Analogous to [`std::io::BufRead::fill_buf`], this should
    /// be paired with [`FlacSampleReader::consume`] to
    /// consume samples in the filled buffer once used.
    ///
    /// Returned samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// # Errors
    ///
    /// Returns error if some error occurs reading FLAC file
    /// to fill buffer.
    pub fn fill_buf(&mut self) -> Result<&[i32], Error> {
        if self.buf.is_empty() {
            match self.decoder.read_frame()? {
                Some(frame) => {
                    self.buf.extend(frame.iter());
                }
                None => return Ok(&[]),
            }
        }

        Ok(self.buf.make_contiguous())
    }

    /// Informs the reader that `amt` samples have been consumed.
    ///
    /// Analagous to [`std::io::BufRead::consume`], which marks
    /// samples as having been read.
    ///
    /// May panic if attempting to consume more bytes
    /// than are available in the buffer.
    pub fn consume(&mut self, amt: usize) {
        self.buf.drain(0..amt);
    }
}

impl FlacSampleReader<BufReader<File>> {
    /// Opens FLAC file from the given path
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        FlacSampleReader::new(BufReader::new(File::open(path.as_ref())?))
    }
}

impl<R: std::io::Read + std::io::Seek> FlacSampleReader<R> {
    /// Seeks to the given channel-independent sample
    ///
    /// The sample is relative to the beginning of the stream
    pub fn seek(&mut self, sample: u64) -> Result<(), Error> {
        let mut pos = self.decoder.seek(sample)?;

        // seeking invalidates the current buffer
        self.buf.clear();

        while pos < sample {
            let buf = self.fill_buf()?;
            let to_consume = buf.len().min((sample - pos) as usize);
            pos += to_consume as u64;
            self.consume(to_consume);
        }

        Ok(())
    }
}

/// A FLAC reader which operates on streamed input
///
/// Because this reader needs to scan the stream for
/// valid frame sync codes before playback,
/// it requires [`std::io::BufRead`] instead of [`std::io::Read`].
///
/// # Example
///
/// ```
/// use flac_codec::{
///     decode::{FlacStreamReader, FrameBuf},
///     encode::{FlacStreamWriter, EncodingOptions},
/// };
/// use std::io::{Cursor, Seek};
/// use std::num::NonZero;
/// use bitstream_io::SignedBitCount;
///
/// let mut flac = Cursor::new(vec![]);
///
/// let samples = (0..100).collect::<Vec<i32>>();
///
/// let mut w = FlacStreamWriter::new(&mut flac, EncodingOptions::default());
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
pub struct FlacStreamReader<R> {
    // the wrapped reader
    reader: R,
    // raw decoded frame samples
    buf: Frame,
    // interlaced frame samples
    samples: Vec<i32>,
}

impl<R: std::io::BufRead> FlacStreamReader<R> {
    /// Opens new FLAC stream reader which wraps the given reader
    #[inline]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: Frame::default(),
            samples: Vec::default(),
        }
    }

    /// Returns the next decoded FLAC frame and its parameters
    ///
    /// # Errors
    ///
    /// Returns an I/O error from the stream or if any
    /// other error occurs when reading the file.
    pub fn read(&mut self) -> Result<FrameBuf<'_>, Error> {
        use crate::crc::{Checksum, Crc16, CrcReader};
        use crate::stream::FrameHeader;
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        // Finding the next frame header in a BufRead is
        // tougher than it seems because fill_buf might
        // slice a frame sync code in half, which needs
        // to be accounted for.

        let (header, mut crc16_reader) = loop {
            // scan for the first byte of the frame sync
            self.reader.skip_until(0b11111111)?;

            // either gotten the first half of the frame sync,
            // or have reached EOF

            // check that the next byte is the other half of a frame sync
            match self.reader.fill_buf() {
                Ok([]) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "eof looking for frame sync",
                    )
                    .into());
                }
                Ok([byte, ..]) if byte >> 1 == 0b1111100 => {
                    // got a whole frame sync
                    // so try to parse a whole frame header
                    let mut crc_reader: CrcReader<_, Crc16> = CrcReader::new(
                        std::slice::from_ref(&0b11111111).chain(self.reader.by_ref()),
                    );

                    if let Ok(header) = FrameHeader::read_subset(&mut crc_reader) {
                        break (header, crc_reader);
                    }
                }
                Ok(_) => continue,
                // didn't get the other half of frame sync,
                // so continue without consuming anything
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        };

        read_subframes(
            BitReader::endian(crc16_reader.by_ref(), BigEndian),
            &header,
            &mut self.buf,
        )?;

        if crc16_reader.into_checksum().valid() {
            self.samples.clear();
            self.samples.extend(self.buf.iter());

            Ok(FrameBuf {
                samples: self.samples.as_slice(),
                sample_rate: header.sample_rate.into(),
                channels: header.channel_assignment.count(),
                bits_per_sample: header.bits_per_sample.into(),
            })
        } else {
            Err(Error::Crc16Mismatch)
        }
    }
}

/// A buffer of samples read from a [`FlacStreamReader`]
///
/// In a conventional FLAC reader, the stream's metadata
/// is known in advance from the required STREAMINFO metadata block
/// and is an error for it to change mid-file.
///
/// In a streamed reader, that metadata isn't known in advance
/// and can change from frame to frame.  This buffer contains
/// all the metadata fields in the frame for decoding/playback.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FrameBuf<'s> {
    /// Decoded samples
    ///
    /// Samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    pub samples: &'s [i32],

    /// The sample rate, in Hz
    pub sample_rate: u32,

    /// Channel count, from 1 to 8
    pub channels: u8,

    /// Bits-per-sample, from 4 to 32
    pub bits_per_sample: u32,
}

/// A FLAC decoder
struct Decoder<R> {
    reader: R,
    // all metadata blocks
    blocks: BlockList,
    // the size of everything before the first frame, in bytes
    frames_start: u64,
    // the current sample, in channel-independent samples
    current_sample: u64,
    // raw decoded frame samples
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
    fn new(mut reader: R) -> Result<Self, Error> {
        use crate::Counter;
        use crate::metadata::read_blocks;
        use std::io::Read;

        let mut counter = Counter::new(reader.by_ref());

        let blocks = read_blocks(counter.by_ref()).collect::<Result<Result<_, _>, _>>()??;

        Ok(Self {
            frames_start: counter.count,
            reader,
            current_sample: 0,
            blocks,
            buf: Frame::default(),
        })
    }

    /// Returns channel count
    ///
    /// From 1 to 8
    fn channel_count(&self) -> NonZero<u8> {
        self.blocks.streaminfo().channels
    }

    /// Returns sample rate, in Hz
    fn sample_rate(&self) -> u32 {
        self.blocks.streaminfo().sample_rate
    }

    /// Returns decoder's bits-per-sample
    ///
    /// From 1 to 32
    fn bits_per_sample(&self) -> u32 {
        self.blocks.streaminfo().bits_per_sample.into()
    }

    /// Returns total number of channel-independent samples, if known
    fn total_samples(&self) -> Option<NonZero<u64>> {
        self.blocks.streaminfo().total_samples
    }

    /// Returns MD5 of entire stream, if known
    fn md5(&self) -> Option<&[u8; 16]> {
        self.blocks.streaminfo().md5.as_ref()
    }

    /// Returns iterator over all metadata blocks
    fn metadata(&self) -> impl Iterator<Item = BlockRef<'_>> {
        self.blocks.blocks()
    }

    /// Returns decoded frame, if any.
    ///
    /// # Errors
    ///
    /// Returns any decoding error from the stream.
    fn read_frame(&mut self) -> Result<Option<&Frame>, Error> {
        use crate::crc::{Checksum, Crc16, CrcReader};
        use crate::stream::FrameHeader;
        use bitstream_io::{BigEndian, BitReader};
        use std::io::Read;

        let mut crc16_reader: CrcReader<_, Crc16> = CrcReader::new(self.reader.by_ref());

        let header = match self
            .blocks
            .streaminfo()
            .total_samples
            .map(|total| total.get() - self.current_sample)
        {
            Some(0) => return Ok(None),
            Some(remaining) => FrameHeader::read(crc16_reader.by_ref(), self.blocks.streaminfo())
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
            None => match FrameHeader::read(crc16_reader.by_ref(), self.blocks.streaminfo()) {
                Ok(header) => header,
                Err(Error::Io(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(None);
                }
                Err(err) => return Err(err),
            },
        };

        read_subframes(
            BitReader::endian(crc16_reader.by_ref(), BigEndian),
            &header,
            &mut self.buf,
        )?;

        if !crc16_reader.into_checksum().valid() {
            return Err(Error::Crc16Mismatch);
        }

        self.current_sample += u64::from(u16::from(header.block_size));

        Ok(Some(&self.buf))
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
    fn seek(&mut self, sample: u64) -> Result<u64, Error> {
        use crate::metadata::SeekPoint;
        use std::io::SeekFrom;

        match self.blocks.get() {
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
                        self.current_sample = *sample_offset;
                        Ok(*sample_offset)
                    }
                    _ => {
                        // empty seektable so rewind to start of stream
                        self.reader.seek(SeekFrom::Start(self.frames_start))?;
                        self.current_sample = 0;
                        Ok(0)
                    }
                }
            }
            None => {
                // no seektable
                // all we can do is rewind data to start of stream
                self.reader.seek(SeekFrom::Start(self.frames_start))?;
                self.current_sample = 0;
                Ok(0)
            }
        }
    }
}

fn read_subframes<R: BitRead>(
    mut reader: R,
    header: &crate::stream::FrameHeader,
    buf: &mut Frame,
) -> Result<(), Error> {
    use crate::stream::ChannelAssignment;

    match header.channel_assignment {
        ChannelAssignment::Independent(total_channels) => {
            buf.resized_channels(
                header.bits_per_sample.into(),
                total_channels.into(),
                u16::from(header.block_size).into(),
            )
            .try_for_each(|channel| {
                read_subframe(&mut reader, header.bits_per_sample.into(), channel)
            })?;
        }
        ChannelAssignment::LeftSide => {
            let (left, side) = buf.resized_stereo(
                header.bits_per_sample.into(),
                u16::from(header.block_size).into(),
            );

            read_subframe(&mut reader, header.bits_per_sample.into(), left)?;

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

            read_subframe(&mut reader, header.bits_per_sample.into(), right)?;

            side.iter_mut().zip(right.iter()).for_each(|(side, right)| {
                *side += *right;
            });
        }
        ChannelAssignment::MidSide => {
            let (mid, side) = buf.resized_stereo(
                header.bits_per_sample.into(),
                u16::from(header.block_size).into(),
            );

            read_subframe(&mut reader, header.bits_per_sample.into(), mid)?;

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

    Ok(())
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
            let (partition, next) = residuals
                .split_at_mut_checked(
                    (block_size / partition_count)
                        .checked_sub(if p == 0 { predictor_order } else { 0 })
                        .ok_or(Error::InvalidPartitionOrder)?,
                )
                .ok_or(Error::InvalidPartitionOrder)?;

            match reader.parse()? {
                ResidualPartitionHeader::Standard { rice } => {
                    partition.iter_mut().try_for_each(|s| {
                        let msb = reader.read_unary::<1>()?;
                        let lsb = reader.read_counted::<RICE_MAX, u32>(rice)?;
                        let unsigned = (msb << u32::from(rice)) | lsb;
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
