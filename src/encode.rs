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
use crate::stream::{FrameNumber, SampleRate};
use crate::{Counter, Error};
use bitstream_io::{BitWrite, BitWriter, LittleEndian, SignedBitCount};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::num::NonZero;

const MAX_CHANNELS: usize = 8;

/// FLAC encoding options
pub struct EncodingOptions {
    block_size: u16,
    metadata: BTreeMap<BlockType, BlockSet>,
    seektable_style: Option<SeektableStyle>,
}

enum SeektableStyle {
    // Generate seekpoint every nth amount of samples
    Samples(u64),
}

impl EncodingOptions {
    /// Overrides default encoding block size of 4096 samples
    pub fn block_size(self, block_size: u16) -> Self {
        // TODO - enforce minimum block size
        Self { block_size, ..self }
    }

    /// Adds new PADDING block to metadata
    ///
    /// Files may contain multiple PADDING blocks,
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
    /// Creates new comment block if not already present.
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

    /// Replaces entire VORBIS COMMENT metadata block
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

    /// Add new PICTURE block to metadata
    ///
    /// Files may contain multiple PICTURE blocks,
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

    /// Add new CUESHEET block to metadata
    ///
    /// Files may (theoretically) contain multiple CUESHEET blocks,
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

    /// Add new APPLICATION block to metadata
    ///
    /// Files may contain multiple APPLICATION blocks,
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

    /// Generate SEEKTABLE with the given number of samples between seek points
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

impl Default for EncodingOptions {
    fn default() -> Self {
        Self {
            block_size: 4096,
            metadata: BTreeMap::default(),
            seektable_style: None,
        }
    }
}

/// A FLAC encoder
pub struct Encoder<W: std::io::Write + std::io::Seek> {
    // the writer we're outputting to
    writer: Counter<W>,
    // various encoding options
    options: EncodingOptions,
    // a partial frame of PCM samples, divided by channels and then samples
    partial_frame: Vec<Vec<i32>>,
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
    md5: BitWriter<md5::Context, LittleEndian>,
    // whether the encoder has finalized the file
    finalized: bool,
}

impl<W: std::io::Write + std::io::Seek> Encoder<W> {
    const MAX_SAMPLES: u64 = 68_719_476_736;

    /// Creates new encoder with the given parameters
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
            partial_frame: vec![Vec::new(); streaminfo.channels.get().into()],
            sample_rate: streaminfo
                .sample_rate
                .try_into()
                .expect("invalid sample rate"),
            streaminfo,
            frame_number: FrameNumber::default(),
            samples_written: 0,
            seekpoints: Vec::new(),
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
        struct MultiIterator<I>(SmallVec<[I; MAX_CHANNELS]>);

        impl<I: Iterator> Iterator for MultiIterator<I> {
            type Item = SmallVec<[I::Item; MAX_CHANNELS]>;

            fn next(&mut self) -> Option<Self::Item> {
                let v = self
                    .0
                    .iter_mut()
                    .filter_map(|i| i.next())
                    .collect::<SmallVec<_>>();
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

        // update MD5 calculation
        // TODO - if we're encoding from raw LE PCM bytes already
        // update the MD5 from those bytes and bypass this
        // expensive re-conversion step
        // (it might make sense to implement Write for the encoder)
        frame.iter().try_for_each(|i| {
            self.md5
                .write_signed_counted(self.streaminfo.bits_per_sample, i)?;
            self.md5.byte_align()
        })?;

        if self.partial_frame[0].is_empty()
            && frame.pcm_frames() % self.options.block_size as usize == 0
        {
            // no partial samples in the buffer
            // and the input is a multiple of our block size,
            // so encode whole FLAC frames from our input

            let mut buffers = frame.channels().collect::<Vec<_>>();

            MultiIterator(
                buffers
                    .iter_mut()
                    .map(|b| b.chunks_exact(self.options.block_size as usize))
                    .collect(),
            )
            .try_for_each(|frame| {
                encode_frame(
                    &mut self.writer,
                    &mut self.streaminfo,
                    &mut self.frame_number,
                    self.sample_rate,
                    frame,
                )
            })
        } else {
            // populate partial frame with more samples
            let mut buffers = self
                .partial_frame
                .iter_mut()
                .zip(frame.channels())
                .map(|(partial, new)| {
                    partial.extend(new);
                    partial
                })
                .collect::<Vec<_>>();

            // encode any whole frames in partials
            MultiIterator(
                buffers
                    .iter_mut()
                    .map(|b| b.chunks_exact(self.options.block_size as usize))
                    .collect(),
            )
            .try_for_each(|frame| {
                encode_frame(
                    &mut self.writer,
                    &mut self.streaminfo,
                    &mut self.frame_number,
                    self.sample_rate,
                    frame,
                )
            })?;

            // retain any remaining samples not converted to FLAC frames
            let remainder = buffers[0].len() % self.options.block_size as usize;
            let range = (buffers[0].len() - remainder)..;

            for buffer in buffers {
                buffer.copy_within(range.clone(), 0);
                buffer.truncate(remainder);
            }

            Ok(())
        }
    }

    fn finalize_inner(&mut self) -> Result<(), Error> {
        if !self.finalized {
            use crate::metadata::{AsBlockRef, BlockSet, SeekTable};

            self.finalized = true;

            // output any remaining partial frame
            if !self.partial_frame[0].is_empty() {
                encode_frame(
                    &mut self.writer,
                    &mut self.streaminfo,
                    &mut self.frame_number,
                    self.sample_rate,
                    self.partial_frame.iter().map(|s| s.as_slice()).collect(),
                )?;
            }

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

            self.streaminfo.md5 = Some(self.md5.aligned_writer()?.clone().compute().0);

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

fn encode_frame<W>(
    mut writer: W,
    streaminfo: &mut Streaminfo,
    frame_number: &mut FrameNumber,
    sample_rate: SampleRate<u32>,
    frame: SmallVec<[&[i32]; MAX_CHANNELS]>,
) -> Result<(), Error>
where
    W: std::io::Write,
{
    use crate::Counter;
    use crate::crc::{Crc16, CrcWriter};
    use crate::stream::{ChannelAssignment, FrameHeader};
    use bitstream_io::BigEndian;

    debug_assert!(!frame.is_empty());

    let size = Counter::new(writer.by_ref());
    let mut w: CrcWriter<_, Crc16> = CrcWriter::new(size);

    // TODO - channel assignment may vary
    FrameHeader {
        blocking_strategy: false,
        frame_number: *frame_number,
        block_size: (frame[0].len() as u16).try_into().expect("frame cannot be empty"),
        sample_rate,
        bits_per_sample: streaminfo.bits_per_sample,
        channel_assignment: ChannelAssignment::Independent(frame.len() as u8),
    }
    .write(&mut w, streaminfo)?;

    let mut w = BitWriter::endian(w, BigEndian);

    for channel in frame {
        encode_subframe(w.by_ref(), channel, streaminfo.bits_per_sample)?;
    }

    let crc16: u16 = w.aligned_writer()?.checksum().into();
    w.write_from(crc16)?;

    frame_number.try_increment()?;

    // update minimum and maximum frame size values
    if let s @ Some(size) = u32::try_from(w.into_writer().into_writer().count)
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

fn encode_subframe<W: BitWrite>(
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

    // TODO - try different subframe types

    encode_fixed_subframe(writer, channel, bits_per_sample, wasted_bps)
    // match channel {
    //     [first] => encode_constant_subframe(w, *first, bits_per_sample, wasted_bps),
    //     [first, rest @ ..] if rest.iter().all(|s| s == first) => {
    //         encode_constant_subframe(w, *first, bits_per_sample, wasted_bps)
    //     }
    //     _ => encode_verbatim_subframe(w, channel, bits_per_sample, wasted_bps),
    // }
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
    writer: &mut W,
    channel: &[i32],
    bits_per_sample: SignedBitCount<32>,
    wasted_bps: u32,
) -> Result<(), Error> {
    use crate::stream::{SubframeHeader, SubframeHeaderType};
    use smallvec::smallvec;

    // TODO - reuse buffers between calls
    let mut buffers: SmallVec<[Vec<i32>; 4]> = smallvec![vec![]; 4];

    // calculate residuals for FIXED subframe orders 0-4
    // (or fewer, if we don't have enough samples)
    let (order, warm_up, residuals) = {
        let mut fixed_orders: SmallVec<[&[i32]; 5]> = smallvec![channel; 1];

        // accumulate a set of FIXED diffs
        for buf in buffers.iter_mut() {
            let prev_order = fixed_orders.last().unwrap();
            match prev_order.split_at_checked(1) {
                Some((_, r)) => {
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

    write_residuals(writer, 0, residuals)
}

fn write_residuals<W: BitWrite>(
    writer: &mut W,
    predictor_order: usize,
    residuals: &[i32],
) -> Result<(), Error> {
    use crate::stream::ResidualPartitionHeader;
    use bitstream_io::BitCount;

    const MAX_PARTITIONS: usize = 64;

    struct Partition<'r, const RICE_MAX: u32> {
        header: ResidualPartitionHeader<RICE_MAX>,
        residuals: &'r [i32],
    }

    impl<'r, const RICE_MAX: u32> Partition<'r, RICE_MAX> {
        fn new(partition: &'r [i32], estimated_bits: &mut u32) -> Self {
            debug_assert!(!partition.is_empty());

            let partition_sum = partition.iter().map(|i| i.unsigned_abs()).sum::<u32>();

            match (partition_sum / partition.len() as u32).checked_ilog2() {
                Some(rice) => {
                    let rice = BitCount::try_from(rice).expect("excessive Rice parameters");
                    assert!(u32::from(rice) < u32::from(BitCount::<RICE_MAX>::new::<RICE_MAX>()));

                    // TODO - should double-check this estimated bits calculation
                    *estimated_bits += 4
                        + ((1 + u32::from(rice)) * partition.len() as u32)
                        + (partition_sum >> (u32::from(rice).saturating_sub(1)))
                        + ((partition.len() as u32) >> 1);

                    // TODO - if estimated bits is larger than
                    // a verbatim (escaped) partition,
                    // just escape the residuals instead

                    Partition {
                        header: ResidualPartitionHeader::Standard { rice },
                        residuals: partition,
                    }
                }
                // all partition residuals are 0, so use a constant
                None => Partition {
                    header: ResidualPartitionHeader::Constant,
                    residuals: partition,
                },
            }
        }
    }

    fn best_partitions<const RICE_MAX: u32>(
        block_size: usize,
        residuals: &[i32],
    ) -> SmallVec<[Partition<'_, RICE_MAX>; MAX_PARTITIONS]> {
        (0..=block_size.trailing_zeros().min(6))
            .map(|partition_order| 1 << partition_order)
            .map(|partition_count| {
                let mut estimated_bits = 0;

                let partitions = residuals
                    .rchunks(block_size / partition_count as usize)
                    .rev()
                    .map(|partition| Partition::new(partition, &mut estimated_bits))
                    .collect();

                (partitions, estimated_bits)
            })
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
    writer.write::<2, u8>(0)?;
    write_block::<0b1111, W>(writer, predictor_order, residuals)
}
