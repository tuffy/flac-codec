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
    Application, BlockSet, BlockSize, BlockType, Cuesheet, MetadataBlock, Picture, SeekPoint,
    Streaminfo, VorbisComment, write_blocks,
};
use crate::stream::{ChannelAssignment, FrameNumber, SampleRate};
use crate::{Counter, Error};
use arrayvec::ArrayVec;
use bitstream_io::{BigEndian, BitRecorder, BitWrite, BitWriter, SignedBitCount};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZero;

const MAX_CHANNELS: usize = 8;

/// A FLAC writer which accepts samples as bytes
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
/// use std::num::NonZero;
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// let mut writer = FlacWriter::endian(
///     &mut flac,                   // our wrapped writer
///     LittleEndian,                // .wav-style byte order
///     EncodingOptions::default(),  // default encoding options
///     44100,                       // sample rate
///     16,                          // bits-per-sample
///     NonZero::new(1).unwrap(),    // channel count
///     NonZero::new(1000),          // total samples
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
    /// `sample_rate` must be between 0 (for non-audio streams)
    /// and 1,048,576 (a 20 bit field).
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// `total_samples`, if known, must be between
    /// 1 and 68,719,476,736 (a 36 bit field).
    ///
    /// Note that if `total_samples` is indicated,
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
        options: EncodingOptions,
        sample_rate: u32,
        bits_per_sample: impl TryInto<SignedBitCount<32>>,
        channels: NonZero<u8>,
        total_samples: Option<NonZero<u64>>,
    ) -> Result<Self, Error> {
        let bits_per_sample = bits_per_sample
            .try_into()
            .map_err(|_| Error::InvalidBitsPerSample)?;

        let bytes_per_sample = u32::from(bits_per_sample).div_ceil(8) as usize;

        let pcm_frame_size = bytes_per_sample * channels.get() as usize;

        Ok(Self {
            buf: VecDeque::default(),
            frame: Frame::empty(channels.get().into(), bits_per_sample.into()),
            bytes_per_sample,
            pcm_frame_size,
            frame_byte_size: pcm_frame_size * options.block_size as usize,
            encoder: Encoder::new(
                writer,
                options,
                sample_rate,
                bits_per_sample,
                channels,
                total_samples,
            )?,
            finalized: false,
            endianness: std::marker::PhantomData,
        })
    }

    /// Creates new FLAC writer in the given endianness with the given parameters
    ///
    /// The writer should be positioned at the start of the file.
    ///
    /// `sample_rate` must be between 0 (for non-audio streams)
    /// and 1,048,576 (a 20 bit field).
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// `total_samples`, if known, must be between
    /// 1 and 68,719,476,736 (a 36 bit field).
    ///
    /// Note that if `total_samples` is indicated,
    /// the number of channel-independent samples written *must*
    /// be equal to that amount or an error will occur when writing
    /// or finalizing the stream.
    ///
    /// # Errors
    ///
    /// Returns I/O error if unable to write initial
    /// metadata blocks.
    /// Returns error if any of the encoding parameters are invalid.
    #[inline]
    pub fn endian(
        writer: W,
        _endianness: E,
        options: EncodingOptions,
        sample_rate: u32,
        bits_per_sample: impl TryInto<SignedBitCount<32>>,
        channels: NonZero<u8>,
        total_samples: Option<NonZero<u64>>,
    ) -> Result<Self, Error> {
        Self::new(
            writer,
            options,
            sample_rate,
            bits_per_sample,
            channels,
            total_samples,
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

impl<W: std::io::Write + std::io::Seek, E: crate::byteorder::Endianness> std::io::Write
    for FlacWriter<W, E>
{
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
        // nothing to do since we don't want to flush a partial
        // FLAC frame to disk until we're done with the whole stream
        Ok(())
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
///     NonZero::new(1).unwrap(),    // channel count
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
    /// `sample_rate` must be between 0 (for non-audio streams)
    /// and 1,048,576 (a 20 bit field).
    ///
    /// `bits_per_sample` must be between 1 and 32.
    ///
    /// `channels` must be between 1 and 8.
    ///
    /// `total_samples`, if known, must be between
    /// 1 and 68,719,476,736 (a 36 bit field).
    ///
    /// Note that if `total_samples` is indicated,
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
        options: EncodingOptions,
        sample_rate: u32,
        bits_per_sample: impl TryInto<SignedBitCount<32>>,
        channels: NonZero<u8>,
        total_samples: Option<NonZero<u64>>,
    ) -> Result<Self, Error> {
        let bits_per_sample = bits_per_sample
            .try_into()
            .map_err(|_| Error::InvalidBitsPerSample)?;

        let bytes_per_sample = u32::from(bits_per_sample).div_ceil(8) as usize;

        let pcm_frame_size = usize::from(channels.get());

        Ok(Self {
            sample_buf: VecDeque::default(),
            frame: Frame::empty(channels.get().into(), bits_per_sample.into()),
            bytes_per_sample,
            pcm_frame_size,
            frame_sample_size: pcm_frame_size * options.block_size as usize,
            encoder: Encoder::new(
                writer,
                options,
                sample_rate,
                bits_per_sample,
                channels,
                total_samples,
            )?,
            finalized: false,
        })
    }

    /// Given a set of samples, writes them to the FLAC file
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

/// FLAC encoding options
pub struct EncodingOptions {
    block_size: u16,
    metadata: BTreeMap<BlockType, BlockSet>,
    seektable_style: Option<SeektableStyle>,
}

impl Default for EncodingOptions {
    fn default() -> Self {
        Self {
            block_size: 4096,
            metadata: BTreeMap::default(),
            // TODO - make default seektable style
            // one point every 10 seconds
            seektable_style: None,
        }
    }
}

enum SeektableStyle {
    // Generate seekpoint every nth amount of samples
    Samples(u64),
}

impl EncodingOptions {
    /// Sets new block size
    pub fn block_size(self, block_size: u16) -> Self {
        Self { block_size, ..self }
    }

    /// Adds new [`crate::metadata::Padding`] block to metadata
    ///
    /// Files may contain multiple [`crate::metadata::Padding`] blocks,
    /// and this adds a new block each time it is used.
    ///
    /// The default is to not add any padding to the output file,
    /// which may be inconvenient if one wishes to modify metadata
    /// later since it will likely require rewriting the whole file
    /// instead of only metadata blocks.
    pub fn padding<B: Into<BlockSize>>(mut self, size: B) -> Self {
        use crate::metadata::Padding;

        match self.metadata.entry(Padding::TYPE) {
            Entry::Occupied(o) => {
                match o.into_mut() {
                    BlockSet::Padding(v) => {
                        v.push(Padding { size: size.into() });
                    }
                    _ => {
                        // this shouldn't happen
                        panic!("Padding blockset not associated with Padding type");
                    }
                }
            }
            Entry::Vacant(v) => {
                v.insert(BlockSet::Padding(vec![Padding { size: size.into() }]));
            }
        }

        self
    }

    /// Adds new tag to comment metadata block
    ///
    /// Creates new [`crate::metadata::VorbisComment`] block if not already present.
    pub fn tag<S>(mut self, field: &str, value: S) -> Self
    where
        S: std::fmt::Display,
    {
        match self.metadata.entry(VorbisComment::TYPE) {
            Entry::Occupied(o) => match o.into_mut() {
                BlockSet::VorbisComment(c) => {
                    c.append_field(field, value);
                }
                _ => {
                    panic!("VorbisComment blockset not associated with VorbisComment type")
                }
            },
            Entry::Vacant(v) => {
                let mut comment = VorbisComment::default();
                comment.append_field(field, value);
                v.insert(BlockSet::VorbisComment(comment));
            }
        }

        self
    }

    /// Replaces entire [`crate::metadata::VorbisComment`] metadata block
    ///
    /// This may be more convenient when adding many fields at once.
    pub fn comment(mut self, comment: VorbisComment) -> Self {
        match self.metadata.entry(VorbisComment::TYPE) {
            Entry::Occupied(o) => {
                *o.into_mut() = BlockSet::VorbisComment(comment);
            }
            Entry::Vacant(v) => {
                v.insert(BlockSet::VorbisComment(comment));
            }
        }

        self
    }

    /// Add new [`crate::metadata::Picture`] block to metadata
    ///
    /// Files may contain multiple [`crate::metadata::Picture`] blocks,
    /// and this adds a new block each time it is used.
    pub fn picture(mut self, picture: Picture) -> Self {
        match self.metadata.entry(Picture::TYPE) {
            Entry::Occupied(o) => {
                match o.into_mut() {
                    BlockSet::Picture(v) => {
                        v.push(picture);
                    }
                    _ => {
                        // this shouldn't happen
                        panic!("Picture blockset not associated with Picture type");
                    }
                }
            }
            Entry::Vacant(v) => {
                v.insert(BlockSet::Picture(vec![picture]));
            }
        }

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
        match self.metadata.entry(Cuesheet::TYPE) {
            Entry::Occupied(o) => {
                match o.into_mut() {
                    BlockSet::Cuesheet(v) => {
                        v.push(cuesheet);
                    }
                    _ => {
                        // this shouldn't happen
                        panic!("Cuesheet blockset not associated with Cuesheet type");
                    }
                }
            }
            Entry::Vacant(v) => {
                v.insert(BlockSet::Cuesheet(vec![cuesheet]));
            }
        }

        self
    }

    /// Add new [`crate::metadata::Application`] block to metadata
    ///
    /// Files may contain multiple [`crate::metadata::Application`] blocks,
    /// and this adds a new block each time it is used.
    pub fn application(mut self, application: Application) -> Self {
        match self.metadata.entry(Application::TYPE) {
            Entry::Occupied(o) => {
                match o.into_mut() {
                    BlockSet::Application(v) => {
                        v.push(application);
                    }
                    _ => {
                        // this shouldn't happen
                        panic!("Application blockset not associated with Application type");
                    }
                }
            }
            Entry::Vacant(v) => {
                v.insert(BlockSet::Application(vec![application]));
            }
        }

        self
    }

    /// Generate [`crate::metadata::SeekTable`] with the given number of samples between seek points
    ///
    /// The interval between seek points may be larger than requested
    /// if the encoder's block size is larger than the seekpoint interval.
    pub fn seektable_samples(mut self, samples: u64) -> Self {
        // note that we can't drop a placeholder seektable
        // into the metadata blocks until we know
        // the sample rate and total samples of our stream
        self.seektable_style = Some(SeektableStyle::Samples(samples));
        self
    }
}

#[derive(Default)]
struct EncodingCaches {
    independent: ChannelCache,
    correlated: CorrelationCache,
}

#[derive(Default)]
struct CorrelationCache {
    // the average channel samples
    average_samples: Vec<i32>,
    // the difference channel samples
    difference_samples: Vec<i32>,
    // the left channel
    left: CorrelationChannel,
    // the right channel
    right: CorrelationChannel,
    // the average channel
    average: CorrelationChannel,
    // the difference channel
    difference: CorrelationChannel,
}

#[derive(Default)]
struct CorrelationChannel {
    cache: ChannelCache,
    output: BitRecorder<u32, BigEndian>,
}

#[derive(Default)]
struct ChannelCache {
    fixed: FixedCache,
    fixed_output: BitRecorder<u32, BigEndian>,
}

#[derive(Default)]
struct FixedCache {
    // FIXED subframe buffers, one per order 1-4
    fixed_buffers: [Vec<i32>; 4],
}

/// A FLAC encoder
struct Encoder<W: std::io::Write + std::io::Seek> {
    // the writer we're outputting to
    writer: Counter<W>,
    // various encoding options
    options: EncodingOptions,
    // various encoder caches
    caches: EncodingCaches,
    // our STREAMINFO block information
    streaminfo: Streaminfo,
    // our stream's sample rate
    sample_rate: SampleRate<u32>,
    // the current frame number
    frame_number: FrameNumber,
    // the number of channel-independent samples written
    samples_written: u64,
    // all seekpoints
    seekpoints: Vec<SeekPoint>,
    // our running MD5 calculation
    md5: md5::Context,
    // whether the encoder has finalized the file
    finalized: bool,
}

impl<W: std::io::Write + std::io::Seek> Encoder<W> {
    const MAX_SAMPLES: u64 = 68_719_476_736;

    fn new(
        mut writer: W,
        mut options: EncodingOptions,
        sample_rate: u32,
        bits_per_sample: impl TryInto<SignedBitCount<32>>,
        channels: NonZero<u8>,
        total_samples: Option<NonZero<u64>>,
    ) -> Result<Self, Error> {
        use crate::metadata::AsBlockRef;

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

        // insert a dummy SeekTable to be populated later
        match options.seektable_style {
            Some(SeektableStyle::Samples(samples)) => {
                if let Some(total_samples) = total_samples {
                    use crate::metadata::SeekTable;

                    options.metadata.insert(
                        BlockType::SeekTable,
                        BlockSet::SeekTable(SeekTable {
                            points: vec![
                                SeekPoint {
                                    sample_offset: None,
                                    byte_offset: 0,
                                    frame_samples: 0,
                                };
                                total_samples
                                    .get()
                                    .div_ceil(samples)
                                    .min(total_samples.get().div_ceil(options.block_size.into()))
                                    .try_into()
                                    .unwrap()
                            ],
                        }),
                    );
                }
            }
            None => { /* do nothing */ }
        }

        write_blocks(
            std::iter::once(streaminfo.as_block_ref())
                .chain(options.metadata.values().flat_map(|v| v.iter())),
            writer.by_ref(),
        )?;

        Ok(Self {
            writer: Counter::new(writer),
            options,
            caches: EncodingCaches::default(),
            sample_rate: streaminfo
                .sample_rate
                .try_into()
                .expect("invalid sample rate"),
            streaminfo,
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
        self.seekpoints.push(SeekPoint {
            sample_offset: Some(self.samples_written),
            byte_offset: self.writer.count,
            frame_samples: frame.pcm_frames() as u16,
        });

        // update running total of samples written
        self.samples_written += frame.pcm_frames() as u64;
        if let Some(total_samples) = self.streaminfo.total_samples {
            if self.samples_written > total_samples.get() {
                return Err(Error::ExcessiveTotalSamples);
            }
        }

        encode_frame(
            &mut self.caches,
            &mut self.writer,
            &mut self.streaminfo,
            &mut self.frame_number,
            self.sample_rate,
            frame.channels().collect(),
        )
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            use crate::metadata::{AsBlockRef, BlockSet, SeekTable};

            self.finalized = true;

            // update SEEKTABLE metadata block with final values
            match self.options.seektable_style {
                Some(SeektableStyle::Samples(samples)) => {
                    // a placeholder SEEKTABLE should always be present
                    if let Some(BlockSet::SeekTable(SeekTable { points })) =
                        self.options.metadata.get_mut(&BlockType::SeekTable)
                    {
                        // grab only the seekpoints that span
                        // "samples" boundaries of PCM samples

                        let mut all_points = self.seekpoints.iter();

                        points
                            .iter_mut()
                            .zip(0..)
                            .for_each(|(seektable_point, frame)| {
                                if let Some(point) = all_points.find(|point| {
                                    point.sample_offset.unwrap() + u64::from(point.frame_samples)
                                        > frame * samples
                                }) {
                                    *seektable_point = point.clone();
                                }
                            });
                    }
                }
                None => { /* no seektable, so nothing to do */ }
            }

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

            self.streaminfo.md5 = Some(self.md5.clone().compute().0);

            let writer = self.writer.stream();

            writer.rewind()?;

            write_blocks(
                std::iter::once(self.streaminfo.as_block_ref())
                    .chain(self.options.metadata.values().flat_map(|v| v.iter())),
                writer.by_ref(),
            )
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

fn encode_frame<W>(
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
        [left, right] => {
            let Correlated {
                channel_assignment,
                channels,
            } = correlate_channels(cache, [left, right], streaminfo.bits_per_sample)?;

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

            for channel in channels {
                channel.playback(&mut bw)?;
            }
        }
        frame => {
            // non-stereo frames are always encoded independently

            FrameHeader {
                blocking_strategy: false,
                frame_number: *frame_number,
                block_size: (frame[0].len() as u16)
                    .try_into()
                    .expect("frame cannot be empty"),
                sample_rate,
                bits_per_sample: streaminfo.bits_per_sample.into(),
                channel_assignment: ChannelAssignment::Independent(frame.len() as u8),
            }
            .write(&mut w, streaminfo)?;

            bw = BitWriter::new(w);

            for channel in frame {
                encode_subframe(
                    &mut cache.independent,
                    bw.by_ref(),
                    channel,
                    streaminfo.bits_per_sample,
                )?;
            }
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

struct Correlated<'c> {
    channel_assignment: ChannelAssignment,
    channels: [&'c mut BitRecorder<u32, BigEndian>; 2],
}

fn correlate_channels<'c>(
    EncodingCaches {
        correlated:
            CorrelationCache {
                left:
                    CorrelationChannel {
                        output: left_channel,
                        cache: left_channel_cache,
                    },
                right:
                    CorrelationChannel {
                        output: right_channel,
                        cache: right_channel_cache,
                    },
                average_samples,
                average:
                    CorrelationChannel {
                        output: average,
                        cache: average_cache,
                    },
                difference_samples,
                difference:
                    CorrelationChannel {
                        output: difference,
                        cache: difference_cache,
                    },
            },
        independent,
    }: &'c mut EncodingCaches,
    [left, right]: [&[i32]; 2],
    bits_per_sample: SignedBitCount<32>,
) -> Result<Correlated<'c>, Error> {
    struct EncodeArgs<'a> {
        cache: &'a mut ChannelCache,
        writer: &'a mut BitRecorder<u32, BigEndian>,
        channel: &'a [i32],
        bits_per_sample: SignedBitCount<32>,
    }

    fn mass_encode(args: [EncodeArgs<'_>; 4]) -> Result<(), Error> {
        for EncodeArgs {
            cache,
            writer,
            channel,
            bits_per_sample,
        } in args
        {
            writer.clear();
            encode_subframe(cache, writer, channel, bits_per_sample)?;
        }
        Ok(())
    }

    match bits_per_sample.checked_add(1) {
        Some(difference_bits_per_sample) => {
            // TODO - calculate these in parallel

            average_samples.clear();
            average_samples.extend(left.iter().zip(right).map(|(l, r)| (l + r) >> 1));

            difference_samples.clear();
            difference_samples.extend(left.iter().zip(right).map(|(l, r)| l - r));

            mass_encode([
                EncodeArgs {
                    cache: left_channel_cache,
                    writer: left_channel,
                    channel: left,
                    bits_per_sample,
                },
                EncodeArgs {
                    cache: right_channel_cache,
                    writer: right_channel,
                    channel: right,
                    bits_per_sample,
                },
                EncodeArgs {
                    cache: average_cache,
                    writer: average,
                    channel: average_samples,
                    bits_per_sample,
                },
                EncodeArgs {
                    cache: difference_cache,
                    writer: difference,
                    channel: difference_samples,
                    bits_per_sample: difference_bits_per_sample,
                },
            ])?;

            let left_difference = left_channel.written() + difference.written();
            let difference_right = difference.written() + right_channel.written();
            let average_difference = average.written() + difference.written();
            let independent = left_channel.written() + right_channel.written();

            Ok(
                if left_difference < difference_right
                    && left_difference < average_difference
                    && left_difference < independent
                {
                    Correlated {
                        channel_assignment: ChannelAssignment::LeftSide,
                        channels: [left_channel, difference],
                    }
                } else if difference_right < average_difference && difference_right < independent {
                    Correlated {
                        channel_assignment: ChannelAssignment::SideRight,
                        channels: [difference, right_channel],
                    }
                } else if average_difference < independent {
                    Correlated {
                        channel_assignment: ChannelAssignment::MidSide,
                        channels: [average, difference],
                    }
                } else {
                    Correlated {
                        channel_assignment: ChannelAssignment::Independent(2),
                        channels: [left_channel, right_channel],
                    }
                },
            )
        }
        None => {
            // 32 bps stream, so forego difference channel
            // and encode them both indepedently

            left_channel.clear();
            encode_subframe(independent, left_channel, left, bits_per_sample)?;

            right_channel.clear();
            encode_subframe(independent, right_channel, right, bits_per_sample)?;

            Ok(Correlated {
                channel_assignment: ChannelAssignment::Independent(2),
                channels: [left_channel, right_channel],
            })
        }
    }
}

fn encode_subframe<W: BitWrite>(
    ChannelCache {
        fixed: fixed_cache,
        fixed_output,
    }: &mut ChannelCache,
    writer: &mut W,
    channel: &[i32],
    bits_per_sample: SignedBitCount<32>,
) -> Result<(), Error> {
    const WASTED_MAX: NonZero<u32> = NonZero::new(32).unwrap();

    debug_assert!(!channel.is_empty());

    // determine any wasted bits
    // FIXME - pull this from an external buffer?
    let mut wasted = Vec::new();

    let (channel, bits_per_sample, wasted_bps) =
        match channel.iter().try_fold(WASTED_MAX, |acc, sample| {
            NonZero::new(sample.trailing_zeros()).map(|sample| sample.min(acc))
        }) {
            None => (channel, bits_per_sample, 0),
            Some(WASTED_MAX) => {
                return encode_constant_subframe(writer, channel[0], bits_per_sample, 0);
            }
            Some(wasted_bps) => {
                let wasted_bps = wasted_bps.get();
                wasted.extend(channel.iter().map(|sample| sample >> wasted_bps));
                (
                    wasted.as_slice(),
                    bits_per_sample.checked_sub(wasted_bps).unwrap(),
                    wasted_bps,
                )
            }
        };

    fixed_output.clear();
    encode_fixed_subframe(
        fixed_cache,
        fixed_output,
        channel,
        bits_per_sample,
        wasted_bps,
    )?;

    // TODO - output to LPC subframe
    // TODO - write the smaller of FIXED, LPC, and VERBATIM subframes

    let verbatim_len = channel.len() as u32 * u32::from(bits_per_sample);

    if fixed_output.written() < verbatim_len {
        Ok(fixed_output.playback(writer)?)
    } else {
        encode_verbatim_subframe(writer, channel, bits_per_sample, wasted_bps)
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
        for buf in buffers.iter_mut() {
            let prev_order = fixed_orders.last().unwrap();
            match prev_order.split_at_checked(1) {
                Some((_, r)) => {
                    buf.clear();
                    buf.extend(
                        r.iter()
                            .zip(*prev_order)
                            .map(|(n, p)| n - p)
                            .collect::<Vec<_>>(),
                    );
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
                    .map(|r| r.unsigned_abs())
                    .sum::<u32>()
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

    write_residuals(writer, order.into(), residuals)
}

fn write_residuals<W: BitWrite>(
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
        fn new(partition: &'r [i32], estimated_bits: &mut u32) -> Self {
            debug_assert!(!partition.is_empty());

            let partition_samples = partition.len() as u32;
            let partition_sum = partition.iter().map(|i| i.unsigned_abs()).sum::<u32>();

            if partition_sum > 0 {
                let rice = if partition_sum > partition_samples {
                    BitCount::try_from(
                        ((partition_sum as f32) / (partition_samples as f32))
                            .log2()
                            .ceil() as u32,
                    )
                    .expect("excessive Rice parameters")
                } else {
                    BitCount::new::<0>()
                };

                debug_assert!(u32::from(rice) < u32::from(BitCount::<RICE_MAX>::new::<RICE_MAX>()));

                let partition_size: u32 = 4u32
                    + ((1 + u32::from(rice)) * partition_samples)
                    + if u32::from(rice) > 0 {
                        partition_sum >> (u32::from(rice) - 1)
                    } else {
                        partition_sum << 1
                    }
                    - (partition_samples / 2);

                *estimated_bits += partition_size;

                // TODO - if estimated bits is larger than
                // a verbatim (escaped) partition,
                // just escape the residuals instead

                Partition {
                    header: ResidualPartitionHeader::Standard { rice },
                    residuals: partition,
                }
            } else {
                // all partition residuals are 0, so use a constant
                Partition {
                    header: ResidualPartitionHeader::Constant,
                    residuals: partition,
                }
            }
        }
    }

    fn best_partitions<const RICE_MAX: u32>(
        block_size: usize,
        residuals: &[i32],
    ) -> ArrayVec<Partition<'_, RICE_MAX>, MAX_PARTITIONS> {
        // TODO - make max partition order configurable
        (0..=block_size.trailing_zeros().min(6))
            .map(|partition_order| 1 << partition_order)
            .map(|partition_count| {
                let mut estimated_bits = 0;

                let partitions: ArrayVec<_, MAX_PARTITIONS> = residuals
                    .rchunks(block_size / partition_count as usize)
                    .rev()
                    .map(|partition| Partition::new(partition, &mut estimated_bits))
                    .collect();

                (partitions, estimated_bits)
            })
            .take_while(|(partitions, _)| partitions.len().is_power_of_two())
            .min_by_key(|(_, estimated_bits)| *estimated_bits)
            .map(|(partitions, _)| partitions)
            .expect("no best set of partitions found")
    }

    fn write_block<const RICE_MAX: u32, W: BitWrite>(
        writer: &mut W,
        predictor_order: usize,
        residuals: &[i32],
    ) -> Result<(), Error> {
        let block_size = predictor_order + residuals.len();

        let partitions = best_partitions::<RICE_MAX>(block_size, residuals);
        debug_assert!(!partitions.is_empty());
        debug_assert!(partitions.len().is_power_of_two());

        writer.write::<4, u32>(partitions.len().ilog2())?; // partition order

        for Partition { header, residuals } in partitions {
            writer.build(&header)?;
            match header {
                ResidualPartitionHeader::Standard { rice } => {
                    let shift = 1 << u32::from(rice);

                    residuals.iter().try_for_each(|s| {
                        let unsigned = if s.is_negative() {
                            ((-*s as u32 - 1) << 1) + 1
                        } else {
                            (*s as u32) << 1
                        };
                        let (quot, rem) = (unsigned / shift, unsigned % shift);
                        writer.write_unary::<1>(quot)?;
                        writer.write_counted(rice, rem)
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
    write_block::<0b1111, W>(writer, predictor_order, residuals)
}
