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
use crate::metadata::{BlockList, SeekTable};
use bitstream_io::{BitRead, SignedBitCount};
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::num::NonZero;
use std::path::Path;

pub use crate::metadata::Metadata;

trait SignedInteger:
    bitstream_io::SignedInteger + Into<i64> + std::ops::AddAssign + std::ops::Neg<Output = Self>
{
    fn from_i64(i: i64) -> Self;

    fn from_u32(u: u32) -> Self;
}

impl SignedInteger for i32 {
    #[inline(always)]
    fn from_i64(i: i64) -> i32 {
        i as i32
    }

    #[inline(always)]
    fn from_u32(u: u32) -> i32 {
        u as i32
    }
}

impl SignedInteger for i64 {
    #[inline(always)]
    fn from_i64(i: i64) -> i64 {
        i
    }

    #[inline(always)]
    fn from_u32(u: u32) -> i64 {
        u as i64
    }
}

/// A `Read`-like trait for signed integer samples
pub trait FlacSampleRead {
    /// Attempts to fill the buffer with samples and returns quantity read
    ///
    /// Returned samples are interleaved by channel, like:
    /// [left₀ , right₀ , left₁ , right₁ , left₂ , right₂ , …]
    ///
    /// # Errors
    ///
    /// Returns error if some error occurs reading FLAC file
    fn read(&mut self, samples: &mut [i32]) -> Result<usize, Error>;

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
    fn fill_buf(&mut self) -> Result<&[i32], Error>;

    /// Informs the reader that `amt` samples have been consumed.
    ///
    /// Analagous to [`std::io::BufRead::consume`], which marks
    /// samples as having been read.
    ///
    /// May panic if attempting to consume more bytes
    /// than are available in the buffer.
    fn consume(&mut self, amt: usize);
}

/// A FLAC reader which outputs PCM samples as bytes
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
#[derive(Clone)]
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
    /// FLAC stream.  If the file has non-FLAC data
    /// at the beginning (such as ID3v2 tags), one
    /// should skip such data before initializing a `FlacReader`.
    #[inline]
    pub fn new(mut reader: R) -> Result<Self, Error> {
        let blocklist = BlockList::read(reader.by_ref())?;

        Ok(Self {
            decoder: Decoder::new(reader, blocklist),
            buf: VecDeque::default(),
            endianness: std::marker::PhantomData,
        })
    }

    /// Opens new FLAC reader in the given endianness
    ///
    /// The reader must be positioned at the start of the
    /// FLAC stream.  If the file has non-FLAC data
    /// at the beginning (such as ID3v2 tags), one
    /// should skip such data before initializing a `FlacReader`.
    #[inline]
    pub fn endian(reader: R, _endian: E) -> Result<Self, Error> {
        Self::new(reader)
    }

    /// Returns FLAC metadata blocks
    #[inline]
    pub fn metadata(&self) -> &BlockList {
        self.decoder.metadata()
    }
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> Metadata for FlacReader<R, E> {
    #[inline]
    fn channel_count(&self) -> u8 {
        self.decoder.channel_count().get()
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.decoder.sample_rate()
    }

    #[inline]
    fn bits_per_sample(&self) -> u32 {
        self.decoder.bits_per_sample()
    }

    #[inline]
    fn total_samples(&self) -> Option<u64> {
        self.decoder.total_samples().map(|s| s.get())
    }

    #[inline]
    fn md5(&self) -> Option<&[u8; 16]> {
        self.decoder.md5()
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

/// A FLAC reader which outputs PCM samples as signed integers
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
#[derive(Clone)]
pub struct FlacSampleReader<R> {
    // the wrapped decoder
    decoder: Decoder<R>,
    // decoded sample buffer
    buf: VecDeque<i32>,
}

impl<R: std::io::Read> FlacSampleReader<R> {
    /// Opens new FLAC reader which wraps the given reader
    ///
    /// The reader must be positioned at the start of the
    /// FLAC stream.  If the file has non-FLAC data
    /// at the beginning (such as ID3v2 tags), one
    /// should skip such data before initializing a `FlacReader`.
    #[inline]
    pub fn new(mut reader: R) -> Result<Self, Error> {
        let blocklist = BlockList::read(reader.by_ref())?;

        Ok(Self {
            decoder: Decoder::new(reader, blocklist),
            buf: VecDeque::default(),
        })
    }

    /// Returns FLAC metadata blocks
    #[inline]
    pub fn metadata(&self) -> &BlockList {
        self.decoder.metadata()
    }
}

impl FlacSampleReader<BufReader<File>> {
    /// Opens FLAC file from the given path
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        FlacSampleReader::new(BufReader::new(File::open(path.as_ref())?))
    }
}

impl<R: std::io::Read> Metadata for FlacSampleReader<R> {
    #[inline]
    fn channel_count(&self) -> u8 {
        self.decoder.channel_count().get()
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.decoder.sample_rate()
    }

    #[inline]
    fn bits_per_sample(&self) -> u32 {
        self.decoder.bits_per_sample()
    }

    #[inline]
    fn total_samples(&self) -> Option<u64> {
        self.decoder.total_samples().map(|s| s.get())
    }

    #[inline]
    fn md5(&self) -> Option<&[u8; 16]> {
        self.decoder.md5()
    }
}

impl<R: std::io::Read> FlacSampleRead for FlacSampleReader<R> {
    fn read(&mut self, samples: &mut [i32]) -> Result<usize, Error> {
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

    fn fill_buf(&mut self) -> Result<&[i32], Error> {
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

    fn consume(&mut self, amt: usize) {
        self.buf.drain(0..amt);
    }
}

/// A seekable FLAC reader which outputs PCM samples as bytes
///
/// This has an additional [`std::io::Seek`] bound over
/// the wrapped reader in order to enable seeking.
///
/// # Example
///
/// ```
/// use flac_codec::{
///     byteorder::LittleEndian,
///     encode::{FlacWriter, Options},
///     decode::SeekableFlacReader,
/// };
/// use std::io::{Cursor, Read, Seek, SeekFrom, Write};
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
/// let mut reader = SeekableFlacReader::endian(flac, LittleEndian).unwrap();
///
/// // read 2000 bytes
/// let mut read_bytes_1 = vec![];
/// assert!(reader.read_to_end(&mut read_bytes_1).is_ok());
///
/// // ensure input and output matches
/// assert_eq!(read_bytes_1, written_bytes);
///
/// // rewind reader to halfway through file
/// assert!(reader.seek(SeekFrom::Start(1000)).is_ok());
///
/// // read 1000 bytes
/// let mut read_bytes_2 = vec![];
/// assert!(reader.read_to_end(&mut read_bytes_2).is_ok());
///
/// // ensure output matches back half of input
/// assert_eq!(read_bytes_2.len(), 1000);
/// assert!(written_bytes.ends_with(&read_bytes_2));
/// ```
#[derive(Clone)]
pub struct SeekableFlacReader<R, E> {
    // our wrapped FLAC reader
    reader: FlacReader<R, E>,
    // start of first frame from known start of stream
    frames_start: u64,
}

impl<R: std::io::Read + std::io::Seek, E: crate::byteorder::Endianness> SeekableFlacReader<R, E> {
    /// Opens new seekable FLAC reader which wraps the given reader
    ///
    /// The reader must be positioned at the start of the
    /// FLAC stream.  If the file has non-FLAC data
    /// at the beginning (such as ID3v2 tags), one
    /// should skip such data before initializing a `FlacReader`.
    #[inline]
    pub fn new(mut reader: R) -> Result<Self, Error> {
        let blocklist = BlockList::read(reader.by_ref())?;
        let frames_start = reader.stream_position()?;

        Ok(Self {
            frames_start,
            reader: FlacReader {
                decoder: Decoder::new(reader, blocklist),
                buf: VecDeque::default(),
                endianness: std::marker::PhantomData,
            },
        })
    }

    /// Opens new seekable FLAC reader in the given endianness
    ///
    /// The reader must be positioned at the start of the
    /// FLAC stream.  If the file has non-FLAC data
    /// at the beginning (such as ID3v2 tags), one
    /// should skip such data before initializing a `FlacReader`.
    #[inline]
    pub fn endian(reader: R, _endian: E) -> Result<Self, Error> {
        Self::new(reader)
    }

    /// Returns FLAC metadata blocks
    #[inline]
    pub fn metadata(&self) -> &BlockList {
        self.reader.decoder.metadata()
    }
}

impl<E: crate::byteorder::Endianness> SeekableFlacReader<BufReader<File>, E> {
    /// Opens seekable FLAC file from the given path
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P, _endianness: E) -> Result<Self, Error> {
        SeekableFlacReader::new(BufReader::new(File::open(path.as_ref())?))
    }
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> std::io::Read for SeekableFlacReader<R, E> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> std::io::BufRead
    for SeekableFlacReader<R, E>
{
    #[inline]
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.reader.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.reader.consume(amt);
    }
}

impl<R: std::io::Read, E: crate::byteorder::Endianness> Metadata for SeekableFlacReader<R, E> {
    fn channel_count(&self) -> u8 {
        self.reader.channel_count()
    }

    fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }

    fn bits_per_sample(&self) -> u32 {
        self.reader.bits_per_sample()
    }

    fn total_samples(&self) -> Option<u64> {
        self.reader.total_samples()
    }

    fn md5(&self) -> Option<&[u8; 16]> {
        self.reader.md5()
    }
}

impl<R: std::io::Read + std::io::Seek, E: crate::byteorder::Endianness> std::io::Seek
    for SeekableFlacReader<R, E>
{
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        use std::cmp::Ordering;

        let FlacReader { decoder, buf, .. } = &mut self.reader;

        let streaminfo = decoder.blocks.streaminfo();

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
                    let original_pos: u64 =
                        (decoder.current_sample * bytes_per_pcm_frame) - (buf.len() as u64);

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
                    let max_pos: u64 =
                        decoder
                            .total_samples()
                            .map(|s| s.get())
                            .ok_or(std::io::Error::new(
                                std::io::ErrorKind::NotSeekable,
                                "total samples not known",
                            ))?;

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
        let mut new_pos = decoder.seek(self.frames_start, desired_pos / bytes_per_pcm_frame)?
            * bytes_per_pcm_frame;

        // seeking invalidates current buffer
        buf.clear();

        // skip bytes to reach desired sample
        while new_pos < desired_pos {
            use std::io::BufRead;

            let buf = self.reader.fill_buf()?;

            if !buf.is_empty() {
                let to_skip = (usize::try_from(desired_pos - new_pos).unwrap()).min(buf.len());
                self.reader.consume(to_skip);
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

/// A seekable FLAC reader which outputs PCM samples as signed integers
///
/// # Example
///
/// ```
/// use flac_codec::{
///     encode::{FlacSampleWriter, Options},
///     decode::{SeekableFlacSampleReader, FlacSampleRead},
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
/// let mut reader = SeekableFlacSampleReader::new(flac).unwrap();
///
/// // read 1000 samples
/// let mut read_samples_1 = vec![0; 1000];
/// assert!(matches!(reader.read(&mut read_samples_1), Ok(1000)));
///
/// // ensure they match
/// assert_eq!(read_samples_1, written_samples);
///
/// // rewind reader to halfway through file
/// assert!(reader.seek(500).is_ok());
///
/// // read 500 samples
/// let mut read_samples_2 = vec![0; 500];
/// assert!(matches!(reader.read(&mut read_samples_2), Ok(500)));
///
/// // ensure output matches back half of input
/// assert_eq!(read_samples_2.len(), 500);
/// assert!(written_samples.ends_with(&read_samples_2));
/// ```
#[derive(Clone)]
pub struct SeekableFlacSampleReader<R> {
    // the wrapped sample reader
    reader: FlacSampleReader<R>,
    // the start of the FLAC frames, in bytes
    frames_start: u64,
}

impl<R: std::io::Read + std::io::Seek> SeekableFlacSampleReader<R> {
    /// Opens new seekable FLAC reader which wraps the given reader
    pub fn new(mut reader: R) -> Result<Self, Error> {
        let blocklist = BlockList::read(reader.by_ref())?;
        let frames_start = reader.stream_position()?;

        Ok(Self {
            frames_start,
            reader: FlacSampleReader {
                decoder: Decoder::new(reader, blocklist),
                buf: VecDeque::default(),
            },
        })
    }

    /// Returns FLAC metadata blocks
    #[inline]
    pub fn metadata(&self) -> &BlockList {
        self.reader.decoder.metadata()
    }
}

impl SeekableFlacSampleReader<BufReader<File>> {
    /// Opens seekable FLAC file from the given path
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        SeekableFlacSampleReader::new(BufReader::new(File::open(path.as_ref())?))
    }
}

impl<R: std::io::Read> Metadata for SeekableFlacSampleReader<R> {
    fn channel_count(&self) -> u8 {
        self.reader.channel_count()
    }

    fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }

    fn bits_per_sample(&self) -> u32 {
        self.reader.bits_per_sample()
    }

    fn total_samples(&self) -> Option<u64> {
        self.reader.total_samples()
    }

    fn md5(&self) -> Option<&[u8; 16]> {
        self.reader.md5()
    }
}

impl<R: std::io::Read> FlacSampleRead for SeekableFlacSampleReader<R> {
    #[inline]
    fn read(&mut self, samples: &mut [i32]) -> Result<usize, Error> {
        self.reader.read(samples)
    }

    #[inline]
    fn fill_buf(&mut self) -> Result<&[i32], Error> {
        self.reader.fill_buf()
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.reader.consume(amt)
    }
}

impl<R: std::io::Read + std::io::Seek> SeekableFlacSampleReader<R> {
    /// Seeks to the given channel-independent sample
    ///
    /// The sample is relative to the beginning of the stream
    pub fn seek(&mut self, sample: u64) -> Result<(), Error> {
        let mut pos = self.reader.decoder.seek(self.frames_start, sample)?;

        // seeking invalidates the current buffer
        self.reader.buf.clear();

        while pos < sample {
            let buf = self.reader.fill_buf()?;
            let to_consume = buf.len().min((sample - pos) as usize);
            pos += to_consume as u64;
            self.reader.consume(to_consume);
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
///     encode::{FlacStreamWriter, Options},
/// };
/// use std::io::{Cursor, Seek};
/// use std::num::NonZero;
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

/// The results of FLAC file verification
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Verified {
    /// FLAC file has MD5 hash and decoded contents match that hash
    MD5Match,
    /// FLAC file has MD5 hash, but decoded contents do not match
    MD5Mismatch,
    /// FLAC file has no MD5 hash, but decodes successfully
    NoMD5,
}

/// Verifies FLAC file for correctness
pub fn verify<P: AsRef<Path>>(p: P) -> Result<Verified, Error> {
    File::open(p.as_ref())
        .map_err(Error::Io)
        .and_then(|r| verify_reader(BufReader::new(r)))
}

/// Verifies FLAC stream for correctness
///
/// The stream must be set to the start of the FLAC data
pub fn verify_reader<R: std::io::Read>(r: R) -> Result<Verified, Error> {
    use crate::byteorder::LittleEndian;

    let mut r = FlacReader::endian(r, LittleEndian)?;
    match r.md5().cloned() {
        Some(flac_md5) => {
            let mut output_md5 = md5::Context::new();
            std::io::copy(&mut r, &mut output_md5)?;
            Ok(if flac_md5 == output_md5.compute().0 {
                Verified::MD5Match
            } else {
                Verified::MD5Mismatch
            })
        }
        None => std::io::copy(&mut r, &mut std::io::sink())
            .map(|_| Verified::NoMD5)
            .map_err(Error::Io),
    }
}

/// A FLAC decoder
#[derive(Clone)]
struct Decoder<R> {
    reader: R,
    // all metadata blocks
    blocks: BlockList,
    // // the size of everything before the first frame, in bytes
    // frames_start: u64,
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
    fn new(reader: R, blocks: BlockList) -> Self {
        Self {
            reader,
            blocks,
            current_sample: 0,
            buf: Frame::default(),
        }
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

    /// Returns FLAC metadata
    fn metadata(&self) -> &BlockList {
        &self.blocks
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
    fn seek(&mut self, frames_start: u64, sample: u64) -> Result<u64, Error> {
        use crate::metadata::SeekPoint;
        use std::io::SeekFrom;

        match self.blocks.get() {
            Some(SeekTable { points: seektable }) => {
                match seektable
                    .iter()
                    .filter(|point| {
                        point
                            .sample_offset()
                            .map(|offset| offset <= sample)
                            .unwrap_or(false)
                    })
                    .next_back()
                {
                    Some(SeekPoint::Defined {
                        sample_offset,
                        byte_offset,
                        ..
                    }) => {
                        assert!(*sample_offset <= sample);
                        self.reader
                            .seek(SeekFrom::Start(frames_start + byte_offset))?;
                        self.current_sample = *sample_offset;
                        Ok(*sample_offset)
                    }
                    _ => {
                        // empty seektable so rewind to start of stream
                        self.reader.seek(SeekFrom::Start(frames_start))?;
                        self.current_sample = 0;
                        Ok(0)
                    }
                }
            }
            None => {
                // no seektable
                // all we can do is rewind data to start of stream
                self.reader.seek(SeekFrom::Start(frames_start))?;
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

            match header.bits_per_sample.checked_add(1) {
                Some(side_bps) => {
                    read_subframe(&mut reader, side_bps, side)?;

                    left.iter().zip(side.iter_mut()).for_each(|(left, side)| {
                        *side = *left - *side;
                    });
                }
                None => {
                    // the very rare case of 32-bps streams
                    // accompanied by side channels
                    let mut side_i64 = vec![0; side.len()];

                    read_subframe::<33, R, i64>(
                        &mut reader,
                        SignedBitCount::from(header.bits_per_sample)
                            .checked_add(1)
                            .expect("excessive bps for substream"),
                        &mut side_i64,
                    )?;

                    left.iter().zip(side_i64).zip(side.iter_mut()).for_each(
                        |((left, side_i64), side)| {
                            *side = (*left as i64 - side_i64) as i32;
                        },
                    );
                }
            }
        }
        ChannelAssignment::SideRight => {
            let (side, right) = buf.resized_stereo(
                header.bits_per_sample.into(),
                u16::from(header.block_size).into(),
            );

            match header.bits_per_sample.checked_add(1) {
                Some(side_bps) => {
                    read_subframe(&mut reader, side_bps, side)?;
                    read_subframe(&mut reader, header.bits_per_sample.into(), right)?;

                    side.iter_mut().zip(right.iter()).for_each(|(side, right)| {
                        *side += *right;
                    });
                }
                None => {
                    // the very rare case of 32-bps streams
                    // accompanied by side channels
                    let mut side_i64 = vec![0; side.len()];

                    read_subframe::<33, R, i64>(
                        &mut reader,
                        SignedBitCount::from(header.bits_per_sample)
                            .checked_add(1)
                            .expect("excessive bps for substream"),
                        &mut side_i64,
                    )?;
                    read_subframe(&mut reader, header.bits_per_sample.into(), right)?;

                    side.iter_mut().zip(side_i64).zip(right.iter()).for_each(
                        |((side, side_64), right)| {
                            *side = (side_64 + *right as i64) as i32;
                        },
                    );
                }
            }
        }
        ChannelAssignment::MidSide => {
            let (mid, side) = buf.resized_stereo(
                header.bits_per_sample.into(),
                u16::from(header.block_size).into(),
            );

            read_subframe(&mut reader, header.bits_per_sample.into(), mid)?;

            match header.bits_per_sample.checked_add(1) {
                Some(side_bps) => {
                    read_subframe(&mut reader, side_bps, side)?;

                    mid.iter_mut().zip(side.iter_mut()).for_each(|(mid, side)| {
                        let sum = *mid * 2 + side.abs() % 2;
                        *mid = (sum + *side) >> 1;
                        *side = (sum - *side) >> 1;
                    });
                }
                None => {
                    // the very rare case of 32-bps streams
                    // accompanied by side channels
                    let mut side_i64 = vec![0; side.len()];

                    read_subframe::<33, R, i64>(
                        &mut reader,
                        SignedBitCount::from(header.bits_per_sample)
                            .checked_add(1)
                            .expect("excessive bps for substream"),
                        &mut side_i64,
                    )?;

                    mid.iter_mut().zip(side.iter_mut()).zip(side_i64).for_each(
                        |((mid, side), side_i64)| {
                            let sum = *mid as i64 * 2 + (side_i64.abs() % 2);
                            *mid = ((sum + side_i64) >> 1) as i32;
                            *side = ((sum - side_i64) >> 1) as i32;
                        },
                    );
                }
            }
        }
    }

    reader.byte_align();
    reader.skip(16)?; // CRC-16 checksum

    Ok(())
}

fn read_subframe<const MAX: u32, R: BitRead, I: SignedInteger>(
    reader: &mut R,
    bits_per_sample: SignedBitCount<MAX>,
    channel: &mut [I],
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};

    let header = reader.parse::<SubframeHeader>()?;

    let effective_bps = bits_per_sample
        .checked_sub::<MAX>(header.wasted_bps)
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

fn read_fixed_subframe<const MAX: u32, R: BitRead, I: SignedInteger>(
    reader: &mut R,
    bits_per_sample: SignedBitCount<MAX>,
    coefficients: &[i64],
    channel: &mut [I],
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

fn read_lpc_subframe<const MAX: u32, R: BitRead, I: SignedInteger>(
    reader: &mut R,
    bits_per_sample: SignedBitCount<MAX>,
    predictor_order: NonZero<u8>,
    channel: &mut [I],
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

fn predict<I: SignedInteger>(coefficients: &[i64], qlp_shift: u32, channel: &mut [I]) {
    for split in coefficients.len()..channel.len() {
        let (predicted, residuals) = channel.split_at_mut(split);

        residuals[0] += I::from_i64(
            predicted
                .iter()
                .rev()
                .zip(coefficients)
                .map(|(x, y)| (*x).into() * y)
                .sum::<i64>()
                >> qlp_shift,
        );
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

fn read_residuals<R: BitRead, I: SignedInteger>(
    reader: &mut R,
    predictor_order: usize,
    residuals: &mut [I],
) -> Result<(), Error> {
    fn read_block<const RICE_MAX: u32, R: BitRead, I: SignedInteger>(
        reader: &mut R,
        predictor_order: usize,
        mut residuals: &mut [I],
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
                            -(I::from_u32(unsigned >> 1)) - I::ONE
                        } else {
                            I::from_u32(unsigned >> 1)
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
                    partition.fill(I::ZERO);
                }
            }

            residuals = next;
        }

        Ok(())
    }

    match reader.read::<2, u8>()? {
        0 => read_block::<0b1111, R, I>(reader, predictor_order, residuals),
        1 => read_block::<0b11111, R, I>(reader, predictor_order, residuals),
        _ => Err(Error::InvalidCodingMethod),
    }
}
