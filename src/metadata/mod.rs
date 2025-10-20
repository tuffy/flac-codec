// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For handling a FLAC file's metadata blocks
//!
//! Many items are capitalized simply because they were capitalized
//! in the original FLAC format documentation.
//!
//! # Metadata Blocks
//!
//! FLAC supports seven different metadata block types
//!
//! | Block Type | Purpose |
//! |-----------:|---------|
//! | [STREAMINFO](`Streaminfo`) | stream information such as sample rate, channel count, etc. |
//! | [PADDING](`Padding`) | empty data which can easily be resized as needed |
//! | [APPLICATION](`Application`) | application-specific data such as foreign RIFF WAVE chunks |
//! | [SEEKTABLE](`SeekTable`) | to allow for more efficient seeking within a FLAC file |
//! | [VORBIS_COMMENT](`VorbisComment`) | textual metadata such as track title, artist name, album name, etc. |
//! | [CUESHEET](`Cuesheet`) | the original disc's layout, for CD images |
//! | [PICTURE](`Picture`) | embedded image files such as cover art |

use crate::Error;
use bitstream_io::{
    BigEndian, BitRead, BitReader, BitWrite, ByteRead, ByteReader, FromBitStream,
    FromBitStreamUsing, FromBitStreamWith, LittleEndian, SignedBitCount, ToBitStream,
    ToBitStreamUsing, write::Overflowed,
};
use std::fs::File;
use std::io::BufReader;
use std::num::NonZero;
use std::path::Path;

/// Types related to the CUESHEET metadata block
pub mod cuesheet;

const FLAC_TAG: &[u8; 4] = b"fLaC";

/// A trait for indicating various pieces of FLAC stream metadata
///
/// This metadata may be necessary for decoding a FLAC file
/// to some other container or an output stream.
pub trait Metadata {
    /// Returns channel count
    ///
    /// From 1 to 8
    fn channel_count(&self) -> u8;

    /// Returns channel mask
    ///
    /// This uses the channel mask defined
    /// in the Vorbis comment, if found, or defaults
    /// to FLAC's default channel assignment if not
    fn channel_mask(&self) -> ChannelMask {
        ChannelMask::from_channels(self.channel_count())
    }

    /// Returns sample rate, in Hz
    fn sample_rate(&self) -> u32;

    /// Returns decoder's bits-per-sample
    ///
    /// From 1 to 32
    fn bits_per_sample(&self) -> u32;

    /// Returns total number of channel-independent samples, if known
    fn total_samples(&self) -> Option<u64> {
        None
    }

    /// Returns MD5 of entire stream, if known
    ///
    /// MD5 is always calculated in terms of little-endian,
    /// signed, byte-aligned values.
    fn md5(&self) -> Option<&[u8; 16]> {
        None
    }

    /// Returns total length of decoded file, in bytes
    fn decoded_len(&self) -> Option<u64> {
        self.total_samples().map(|s| {
            s * u64::from(self.channel_count()) * u64::from(self.bits_per_sample().div_ceil(8))
        })
    }

    /// Returns duration of file
    fn duration(&self) -> Option<std::time::Duration> {
        const NANOS_PER_SEC: u64 = 1_000_000_000;

        let sample_rate = u64::from(self.sample_rate());

        self.total_samples().map(|s| {
            std::time::Duration::new(
                s / sample_rate,
                u32::try_from(((s % sample_rate) * NANOS_PER_SEC) / sample_rate)
                    .unwrap_or_default(),
            )
        })
    }
}

/// A FLAC metadata block header
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 1    | `last` | final metadata block in file |
/// | 7    | `block_type` | type of block |
/// | 24   | `size` | block size, in bytes |
///
/// # Example
/// ```
/// use bitstream_io::{BitReader, BitRead, BigEndian};
/// use flac_codec::metadata::{BlockHeader, BlockType};
///
/// let data: &[u8] = &[0b1_0000000, 0x00, 0x00, 0x22];
/// let mut r = BitReader::endian(data, BigEndian);
/// assert_eq!(
///     r.parse::<BlockHeader>().unwrap(),
///     BlockHeader {
///         last: true,                         // 0b1
///         block_type: BlockType::Streaminfo,  // 0b0000000
///         size: 0x00_00_22u16.into(),         // 0x00, 0x00, 0x22
///     },
/// );
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct BlockHeader {
    /// Whether we are the final block
    pub last: bool,
    /// Our block type
    pub block_type: BlockType,
    /// Our block size, in bytes
    pub size: BlockSize,
}

impl BlockHeader {
    const SIZE: BlockSize = BlockSize((1 + 7 + 24) / 8);
}

/// A type of FLAC metadata block
pub trait MetadataBlock:
    ToBitStream<Error: Into<Error>> + Into<Block> + TryFrom<Block> + Clone
{
    /// The metadata block's type
    const TYPE: BlockType;

    /// Whether the block can occur multiple times in a file
    const MULTIPLE: bool;

    /// Size of block, in bytes, not including header
    fn bytes(&self) -> Option<BlockSize> {
        self.bits::<BlockBits>().ok().map(|b| b.into())
    }

    /// Size of block, in bytes, including block header
    fn total_size(&self) -> Option<BlockSize> {
        self.bytes().and_then(|s| s.checked_add(BlockHeader::SIZE))
    }
}

#[derive(Default)]
struct BlockBits(u32);

impl BlockBits {
    const MAX: u32 = BlockSize::MAX * 8;
}

impl From<u8> for BlockBits {
    fn from(u: u8) -> Self {
        Self(u.into())
    }
}

impl TryFrom<u32> for BlockBits {
    type Error = (); // the error will be replaced later

    fn try_from(u: u32) -> Result<Self, Self::Error> {
        (u <= Self::MAX).then_some(Self(u)).ok_or(())
    }
}

impl TryFrom<usize> for BlockBits {
    type Error = (); // the error will be replaced later

    fn try_from(u: usize) -> Result<Self, Self::Error> {
        u32::try_from(u)
            .map_err(|_| ())
            .and_then(|u| (u <= Self::MAX).then_some(Self(u)).ok_or(()))
    }
}

impl bitstream_io::write::Counter for BlockBits {
    fn checked_add_assign(&mut self, Self(b): Self) -> Result<(), Overflowed> {
        *self = self
            .0
            .checked_add(b)
            .filter(|b| *b <= Self::MAX)
            .map(Self)
            .ok_or(Overflowed)?;
        Ok(())
    }

    fn checked_mul(self, Self(b): Self) -> Result<Self, Overflowed> {
        self.0
            .checked_mul(b)
            .filter(|b| *b <= Self::MAX)
            .map(Self)
            .ok_or(Overflowed)
    }

    fn byte_aligned(&self) -> bool {
        self.0.is_multiple_of(8)
    }
}

impl From<BlockBits> for BlockSize {
    fn from(BlockBits(u): BlockBits) -> Self {
        assert!(u % 8 == 0);
        Self(u / 8)
    }
}

impl BlockHeader {
    fn new<M: MetadataBlock>(last: bool, block: &M) -> Result<Self, Error> {
        fn large_block<E: Into<Error>>(err: E) -> Error {
            match err.into() {
                Error::Io(_) => Error::ExcessiveBlockSize,
                e => e,
            }
        }

        Ok(Self {
            last,
            block_type: M::TYPE,
            size: block.bits::<BlockBits>().map_err(large_block)?.into(),
        })
    }
}

impl FromBitStream for BlockHeader {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        Ok(Self {
            last: r.read::<1, _>()?,
            block_type: r.parse()?,
            size: r.parse()?,
        })
    }
}

impl ToBitStream for BlockHeader {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<1, _>(self.last)?;
        w.build(&self.block_type)?;
        w.build(&self.size)?;
        Ok(())
    }
}

/// A defined FLAC metadata block type
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum BlockType {
    /// The STREAMINFO block
    Streaminfo = 0,
    /// The PADDING block
    Padding = 1,
    /// The APPLICATION block
    Application = 2,
    /// The SEEKTABLE block
    SeekTable = 3,
    /// The VORBIS_COMMENT block
    VorbisComment = 4,
    /// The CUESHEET block
    Cuesheet = 5,
    /// The PICTURE block
    Picture = 6,
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Streaminfo => "STREAMINFO".fmt(f),
            Self::Padding => "PADDING".fmt(f),
            Self::Application => "APPLICATION".fmt(f),
            Self::SeekTable => "SEEKTABLE".fmt(f),
            Self::VorbisComment => "VORBIS_COMMENT".fmt(f),
            Self::Cuesheet => "CUESHEET".fmt(f),
            Self::Picture => "PICTURE".fmt(f),
        }
    }
}

impl FromBitStream for BlockType {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        match r.read::<7, u8>()? {
            0 => Ok(Self::Streaminfo),
            1 => Ok(Self::Padding),
            2 => Ok(Self::Application),
            3 => Ok(Self::SeekTable),
            4 => Ok(Self::VorbisComment),
            5 => Ok(Self::Cuesheet),
            6 => Ok(Self::Picture),
            7..=126 => Err(Error::ReservedMetadataBlock),
            _ => Err(Error::InvalidMetadataBlock),
        }
    }
}

impl ToBitStream for BlockType {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<7, u8>(match self {
            Self::Streaminfo => 0,
            Self::Padding => 1,
            Self::Application => 2,
            Self::SeekTable => 3,
            Self::VorbisComment => 4,
            Self::Cuesheet => 5,
            Self::Picture => 6,
        })
        .map_err(Error::Io)
    }
}

/// A block type for optional FLAC metadata blocks
///
/// This is a subset of [`BlockType`] which contains
/// no STREAMINFO, which is a required block.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum OptionalBlockType {
    /// The PADDING block
    Padding = 1,
    /// The APPLICATION block
    Application = 2,
    /// The SEEKTABLE block
    SeekTable = 3,
    /// The VORBIS_COMMENT block
    VorbisComment = 4,
    /// The CUESHEET block
    Cuesheet = 5,
    /// The PICTURE block
    Picture = 6,
}

impl From<OptionalBlockType> for BlockType {
    fn from(block: OptionalBlockType) -> Self {
        match block {
            OptionalBlockType::Padding => Self::Padding,
            OptionalBlockType::Application => Self::Application,
            OptionalBlockType::SeekTable => Self::SeekTable,
            OptionalBlockType::VorbisComment => Self::VorbisComment,
            OptionalBlockType::Cuesheet => Self::Cuesheet,
            OptionalBlockType::Picture => Self::Picture,
        }
    }
}

/// A 24-bit block size value, with safeguards against overflow
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct BlockSize(u32);

impl BlockSize {
    /// A value of 0
    pub const ZERO: BlockSize = BlockSize(0);

    const MAX: u32 = (1 << 24) - 1;

    /// Our current value as a u32
    fn get(&self) -> u32 {
        self.0
    }
}

impl BlockSize {
    /// Conditionally add `BlockSize` to ourself
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.0
            .checked_add(rhs.0)
            .filter(|s| *s <= Self::MAX)
            .map(Self)
    }

    /// Conditionally subtract `BlockSize` from ourself
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
    }
}

impl std::fmt::Display for BlockSize {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FromBitStream for BlockSize {
    type Error = std::io::Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        r.read::<24, _>().map(Self)
    }
}

impl ToBitStream for BlockSize {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write::<24, _>(self.0)
    }
}

impl From<u8> for BlockSize {
    fn from(u: u8) -> Self {
        Self(u.into())
    }
}

impl From<u16> for BlockSize {
    fn from(u: u16) -> Self {
        Self(u.into())
    }
}

impl TryFrom<usize> for BlockSize {
    type Error = BlockSizeOverflow;

    fn try_from(u: usize) -> Result<Self, Self::Error> {
        u32::try_from(u)
            .map_err(|_| BlockSizeOverflow)
            .and_then(|s| (s <= Self::MAX).then_some(Self(s)).ok_or(BlockSizeOverflow))
    }
}

impl TryFrom<u32> for BlockSize {
    type Error = BlockSizeOverflow;

    fn try_from(u: u32) -> Result<Self, Self::Error> {
        (u <= Self::MAX).then_some(Self(u)).ok_or(BlockSizeOverflow)
    }
}

impl TryFrom<u64> for BlockSize {
    type Error = BlockSizeOverflow;

    fn try_from(u: u64) -> Result<Self, Self::Error> {
        u32::try_from(u)
            .map_err(|_| BlockSizeOverflow)
            .and_then(|s| (s <= Self::MAX).then_some(Self(s)).ok_or(BlockSizeOverflow))
    }
}

impl From<BlockSize> for u32 {
    #[inline]
    fn from(size: BlockSize) -> u32 {
        size.0
    }
}

/// An error that occurs when trying to build an overly large `BlockSize`
#[derive(Copy, Clone, Debug)]
pub struct BlockSizeOverflow;

impl std::error::Error for BlockSizeOverflow {}

impl std::fmt::Display for BlockSizeOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        "value too large for BlockSize".fmt(f)
    }
}

/// An iterator over FLAC metadata blocks
pub struct BlockIterator<R: std::io::Read> {
    reader: R,
    failed: bool,
    tag_read: bool,
    streaminfo_read: bool,
    seektable_read: bool,
    vorbiscomment_read: bool,
    png_read: bool,
    icon_read: bool,
    finished: bool,
}

impl<R: std::io::Read> BlockIterator<R> {
    /// Creates an iterator over something that implements `Read`.
    /// Because this may perform many small reads,
    /// performance is greatly improved by buffering reads
    /// when reading from a raw `File`.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            failed: false,
            tag_read: false,
            streaminfo_read: false,
            seektable_read: false,
            vorbiscomment_read: false,
            png_read: false,
            icon_read: false,
            finished: false,
        }
    }

    fn read_block(&mut self) -> Option<Result<Block, Error>> {
        // like a slighly easier variant of "Take"
        struct LimitedReader<R> {
            reader: R,
            size: usize,
        }

        impl<R: std::io::Read> std::io::Read for LimitedReader<R> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let size = self.size.min(buf.len());
                self.reader.read(&mut buf[0..size]).inspect(|amt_read| {
                    self.size -= amt_read;
                })
            }
        }

        (!self.finished).then(|| {
            BitReader::endian(&mut self.reader, BigEndian)
                .parse()
                .and_then(|header: BlockHeader| {
                    let mut reader = BitReader::endian(
                        LimitedReader {
                            reader: self.reader.by_ref(),
                            size: header.size.get().try_into().unwrap(),
                        },
                        BigEndian,
                    );

                    let block = reader.parse_with(&header)?;

                    match reader.into_reader().size {
                        0 => {
                            self.finished = header.last;
                            Ok(block)
                        }
                        _ => Err(Error::InvalidMetadataBlockSize),
                    }
                })
        })
    }
}

impl<R: std::io::Read> Iterator for BlockIterator<R> {
    type Item = Result<Block, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            // once we hit an error, stop any further reads
            None
        } else if !self.tag_read {
            // "fLaC" tag must come before anything else
            let mut tag = [0; 4];
            match self.reader.read_exact(&mut tag) {
                Ok(()) => match &tag {
                    FLAC_TAG => {
                        self.tag_read = true;
                        self.next()
                    }
                    _ => {
                        self.failed = true;
                        Some(Err(Error::MissingFlacTag))
                    }
                },
                Err(err) => {
                    self.failed = true;
                    Some(Err(Error::Io(err)))
                }
            }
        } else if !self.streaminfo_read {
            // STREAMINFO block must be first in file
            match self.read_block() {
                block @ Some(Ok(Block::Streaminfo(_))) => {
                    self.streaminfo_read = true;
                    block
                }
                _ => {
                    self.failed = true;
                    Some(Err(Error::MissingStreaminfo))
                }
            }
        } else {
            match self.read_block() {
                Some(Ok(Block::Streaminfo(_))) => Some(Err(Error::MultipleStreaminfo)),
                seektable @ Some(Ok(Block::SeekTable(_))) => {
                    if !self.seektable_read {
                        self.seektable_read = true;
                        seektable
                    } else {
                        self.failed = true;
                        Some(Err(Error::MultipleSeekTable))
                    }
                }
                vorbiscomment @ Some(Ok(Block::VorbisComment(_))) => {
                    if !self.vorbiscomment_read {
                        self.vorbiscomment_read = true;
                        vorbiscomment
                    } else {
                        self.failed = true;
                        Some(Err(Error::MultipleVorbisComment))
                    }
                }
                picture @ Some(Ok(Block::Picture(Picture {
                    picture_type: PictureType::Png32x32,
                    ..
                }))) => {
                    if !self.png_read {
                        self.png_read = true;
                        picture
                    } else {
                        self.failed = true;
                        Some(Err(Error::MultiplePngIcon))
                    }
                }
                picture @ Some(Ok(Block::Picture(Picture {
                    picture_type: PictureType::GeneralFileIcon,
                    ..
                }))) => {
                    if !self.icon_read {
                        self.icon_read = true;
                        picture
                    } else {
                        self.failed = true;
                        Some(Err(Error::MultipleGeneralIcon))
                    }
                }
                block @ Some(Err(_)) => {
                    self.failed = true;
                    block
                }
                block => block,
            }
        }
    }
}

/// Returns iterator of blocks from the given reader
///
/// The reader should be positioned at the start of the FLAC
/// file.
///
/// Because this may perform many small reads,
/// using a buffered reader may greatly improve performance
/// when reading from a raw `File`.
///
/// # Example
///
/// ```
/// use flac_codec::{
///     metadata::{read_blocks, Application, Block},
///     encode::{FlacSampleWriter, Options},
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// // add some APPLICATION blocks at encode-time
/// let application_1 = Application {id: 0x1234, data: vec![1, 2, 3, 4]};
/// let application_2 = Application {id: 0x5678, data: vec![5, 6, 7, 8]};
///
/// let options = Options::default()
///     .application(application_1.clone())
///     .application(application_2.clone())
///     .no_padding()
///     .no_seektable();
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,  // our wrapped writer
///     options,    // our encoding options
///     44100,      // sample rate
///     16,         // bits-per-sample
///     1,          // channel count
///     Some(1),    // total samples
/// ).unwrap();
///
/// // write a simple FLAC file
/// writer.write(std::slice::from_ref(&0)).unwrap();
/// writer.finalize().unwrap();
///
/// flac.rewind().unwrap();
///
/// // read blocks from encoded file
/// let blocks = read_blocks(flac)
///     .skip(1)  // skip STREAMINFO block
///     .collect::<Result<Vec<Block>, _>>()
///     .unwrap();
///
/// // ensure they match our APPLICATION blocks
/// assert_eq!(blocks, vec![application_1.into(), application_2.into()]);
/// ```
pub fn read_blocks<R: std::io::Read>(r: R) -> BlockIterator<R> {
    BlockIterator::new(r)
}

/// Returns iterator of blocks from the given path
///
/// # Errors
///
/// Returns any I/O error from opening the path.
/// Note that the iterator itself may return any errors
/// from reading individual blocks.
pub fn blocks<P: AsRef<Path>>(p: P) -> std::io::Result<BlockIterator<BufReader<File>>> {
    File::open(p.as_ref()).map(|f| read_blocks(BufReader::new(f)))
}

/// Returns first instance of the given block from the given reader
///
/// The reader should be positioned at the start of the FLAC
/// file.
///
/// Because this may perform many small reads,
/// using a buffered reader may greatly improve performance
/// when reading from a raw `File`.
///
/// # Example
///
/// ```
/// use flac_codec::{
///     metadata::{read_block, Streaminfo},
///     encode::{FlacSampleWriter, Options},
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac: Cursor<Vec<u8>> = Cursor::new(vec![]);  // a FLAC file in memory
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,           // our wrapped writer
///     Options::default(),  // default encoding options
///     44100,               // sample rate
///     16,                  // bits-per-sample
///     1,                   // channel count
///     Some(1),             // total samples
/// ).unwrap();
///
/// // write a simple FLAC file
/// writer.write(std::slice::from_ref(&0)).unwrap();
/// writer.finalize().unwrap();
///
/// flac.rewind().unwrap();
///
/// // STREAMINFO block must always be present
/// let streaminfo = match read_block::<_, Streaminfo>(flac) {
///     Ok(Some(streaminfo)) => streaminfo,
///     _ => panic!("STREAMINFO not found"),
/// };
///
/// // verify STREAMINFO fields against encoding parameters
/// assert_eq!(streaminfo.sample_rate, 44100);
/// assert_eq!(u32::from(streaminfo.bits_per_sample), 16);
/// assert_eq!(streaminfo.channels.get(), 1);
/// assert_eq!(streaminfo.total_samples.map(|s| s.get()), Some(1));
/// ```
pub fn read_block<R, B>(r: R) -> Result<Option<B>, Error>
where
    R: std::io::Read,
    B: MetadataBlock,
{
    read_blocks(r)
        .find_map(|r| r.map(|b| B::try_from(b).ok()).transpose())
        .transpose()
}

/// Returns first instance of the given block from the given path
///
/// See the [`read_block`] for additional information.
///
/// # Errors
///
/// Returns any error from opening the path, or from reading
/// blocks from the path.
pub fn block<P, B>(p: P) -> Result<Option<B>, Error>
where
    P: AsRef<Path>,
    B: MetadataBlock,
{
    blocks(p).map_err(Error::Io).and_then(|mut blocks| {
        blocks
            .find_map(|r| r.map(|b| B::try_from(b).ok()).transpose())
            .transpose()
    })
}

/// Returns FLAC's STREAMINFO metadata block from the given file
///
/// # Errors
///
/// Returns an error if the STREAMINFO block is not first
/// or if any I/O error occurs when reading the file.
pub fn info<P: AsRef<Path>>(p: P) -> Result<Streaminfo, Error> {
    File::open(p)
        .map_err(Error::Io)
        .and_then(|f| read_info(BufReader::new(f)))
}

/// Returns FLAC's STREAMINFO metadata block from the given reader
/// The reader is assumed to be rewound to the start of the FLAC file data.
///
/// # Errors
///
/// Returns an error if the STREAMINFO block is not first
/// or if any I/O error occurs when reading the file.
pub fn read_info<R: std::io::Read>(r: R) -> Result<Streaminfo, Error> {
    let mut r = BitReader::endian(r, BigEndian);

    // FLAC tag must be first thing in stream
    if &r.read_to::<[u8; 4]>()? != FLAC_TAG {
        return Err(Error::MissingFlacTag);
    }

    // STREAMINFO block must be present, and must be first
    if !matches!(
        r.parse()?,
        BlockHeader {
            block_type: BlockType::Streaminfo,
            size: Streaminfo::SIZE,
            last: _,
        }
    ) {
        return Err(Error::MissingStreaminfo);
    }

    // Finally, parse STREAMINFO block itself
    r.parse().map_err(Error::Io)
}

/// Returns iterator of blocks of a given type
pub fn blocks_of<P, B>(p: P) -> impl Iterator<Item = Result<B, Error>>
where
    P: AsRef<Path>,
    B: MetadataBlock + 'static,
{
    match blocks(p) {
        Ok(iter) => {
            Box::new(iter.filter_map(|block| block.map(|b| B::try_from(b).ok()).transpose()))
                as Box<dyn Iterator<Item = Result<B, Error>>>
        }
        Err(e) => Box::new(std::iter::once(Err(e.into()))),
    }
}

/// Writes iterator of blocks to the given writer.
///
/// Because this may perform many small writes,
/// buffering writes may greatly improve performance
/// when writing to a raw `File`.
///
/// let picture_type = picture.picture_type;
/// # Errorsprintln!("  type: {} ({})", picture_type as u8, picture_type);
///
/// Passes along any I/O errors from the underlying stream.
/// May also generate an error if any of the blocks are invalid
/// (e.g. STREAMINFO not being the first block, any block is too large, etc.).
///
/// # Example
///
/// ```
/// use flac_codec::metadata::{
///     write_blocks, read_blocks, Streaminfo, Application, Block,
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac: Cursor<Vec<u8>> = Cursor::new(vec![]);  // a FLAC file in memory
///
/// // our test blocks
/// let blocks: Vec<Block> = vec![
///     Streaminfo {
///         minimum_block_size: 0,
///         maximum_block_size: 0,
///         minimum_frame_size: None,
///         maximum_frame_size: None,
///         sample_rate: 44100,
///         channels: 1u8.try_into().unwrap(),
///         bits_per_sample: 16u32.try_into().unwrap(),
///         total_samples: None,
///         md5: None,
///     }.into(),
///     Application {id: 0x1234, data: vec![1, 2, 3, 4]}.into(),
///     Application {id: 0x5678, data: vec![5, 6, 7, 8]}.into(),
/// ];
///
/// // write our test blocks to a file
/// write_blocks(&mut flac, blocks.clone()).unwrap();
///
/// flac.rewind().unwrap();
///
/// // read our blocks back from that file
/// let read_blocks = read_blocks(flac).collect::<Result<Vec<Block>, _>>().unwrap();
///
/// // they should be identical
/// assert_eq!(blocks, read_blocks);
/// ```
pub fn write_blocks<B: AsBlockRef>(
    mut w: impl std::io::Write,
    blocks: impl IntoIterator<Item = B>,
) -> Result<(), Error> {
    fn iter_last<T>(i: impl Iterator<Item = T>) -> impl Iterator<Item = (bool, T)> {
        struct LastIterator<I: std::iter::Iterator> {
            iter: std::iter::Peekable<I>,
        }

        impl<T, I: std::iter::Iterator<Item = T>> Iterator for LastIterator<I> {
            type Item = (bool, T);

            fn next(&mut self) -> Option<Self::Item> {
                let item = self.iter.next()?;
                Some((self.iter.peek().is_none(), item))
            }
        }

        LastIterator { iter: i.peekable() }
    }

    // "FlaC" tag must come before anything else
    w.write_all(FLAC_TAG).map_err(Error::Io)?;

    let mut w = bitstream_io::BitWriter::endian(w, BigEndian);
    let mut blocks = iter_last(blocks.into_iter());

    // STREAMINFO block must be present and must be first in file
    let next = blocks.next();
    match next.as_ref().map(|(last, b)| (last, b.as_block_ref())) {
        Some((last, streaminfo @ BlockRef::Streaminfo(_))) => w.build_using(&streaminfo, *last)?,
        _ => return Err(Error::MissingStreaminfo),
    }

    // certain other blocks in the file must only occur once at most
    let mut seektable_read = false;
    let mut vorbiscomment_read = false;
    let mut png_read = false;
    let mut icon_read = false;

    blocks.try_for_each(|(last, block)| match block.as_block_ref() {
        BlockRef::Streaminfo(_) => Err(Error::MultipleStreaminfo),
        vorbiscomment @ BlockRef::VorbisComment(_) => match vorbiscomment_read {
            false => {
                vorbiscomment_read = true;
                w.build_using(&vorbiscomment.as_block_ref(), last)
            }
            true => Err(Error::MultipleVorbisComment),
        },
        seektable @ BlockRef::SeekTable(_) => match seektable_read {
            false => {
                seektable_read = true;
                w.build_using(&seektable.as_block_ref(), last)
            }
            true => Err(Error::MultipleSeekTable),
        },
        picture @ BlockRef::Picture(Picture {
            picture_type: PictureType::Png32x32,
            ..
        }) => {
            if !png_read {
                png_read = true;
                w.build_using(&picture.as_block_ref(), last)
            } else {
                Err(Error::MultiplePngIcon)
            }
        }
        picture @ BlockRef::Picture(Picture {
            picture_type: PictureType::GeneralFileIcon,
            ..
        }) => {
            if !icon_read {
                icon_read = true;
                w.build_using(&picture.as_block_ref(), last)
            } else {
                Err(Error::MultipleGeneralIcon)
            }
        }
        block => w.build_using(&block.as_block_ref(), last),
    })
}

/// Given a Path, attempts to update FLAC metadata blocks
///
/// Returns `true` if the file was completely rebuilt,
/// or `false` if the original was overwritten.
///
/// # Errors
///
/// Returns error if unable to read metadata blocks,
/// unable to write blocks, or if the existing or updated
/// blocks do not conform to the FLAC file specification.
pub fn update<P, E>(path: P, f: impl FnOnce(&mut BlockList) -> Result<(), E>) -> Result<bool, E>
where
    P: AsRef<Path>,
    E: From<Error>,
{
    use std::fs::OpenOptions;

    update_file(
        OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(false)
            .create(false)
            .open(path.as_ref())
            .map_err(Error::Io)?,
        || std::fs::File::create(path.as_ref()),
        f,
    )
}

/// Given open file, attempts to update its metadata blocks
///
/// The original file should be rewound to the start of the stream.
///
/// Applies closure `f` to the blocks and attempts to update them.
///
/// If the updated blocks can be made the same size as the
/// original file by adjusting padding, the file will be
/// partially overwritten with new contents.
///
/// If the new blocks are too large (or small) to fit into
/// the original file, the original unmodified file is dropped
/// and the `rebuilt` closure is called to build a new
/// file.  The file's contents are then dumped into the new file.
///
/// Returns `true` if the file was completely rebuilt,
/// or `false` if the original was overwritten.
///
/// # Example 1
///
/// ```
/// use flac_codec::{
///     metadata::{update_file, BlockList, Padding, VorbisComment},
///     metadata::fields::TITLE,
///     encode::{FlacSampleWriter, Options},
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// // include a small amount of padding
/// const PADDING_SIZE: u32 = 100;
/// let options = Options::default().padding(PADDING_SIZE).unwrap();
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,  // our wrapped writer
///     options,    // our encoding options
///     44100,      // sample rate
///     16,         // bits-per-sample
///     1,          // channel count
///     Some(1),    // total samples
/// ).unwrap();
///
/// // write a simple FLAC file
/// writer.write(std::slice::from_ref(&0)).unwrap();
/// writer.finalize().unwrap();
///
/// flac.rewind().unwrap();
///
/// let mut rebuilt: Vec<u8> = vec![];
///
/// // update file with new Vorbis Comment
/// assert!(matches!(
///     update_file::<_, _, flac_codec::Error>(
///         // our original file
///         &mut flac,
///         // a closure to create a new file, if necessary
///         || Ok(&mut rebuilt),
///         // the closure that performs the metadata update
///         |blocklist| {
///             blocklist.update::<VorbisComment>(
///                 // the blocklist itself has a closure
///                 // that updates a block, creating it if necessary
///                 // (in this case, we're updating the Vorbis comment)
///                 |vc| vc.set(TITLE, "Track Title")
///             );
///             Ok(())
///         },
///     ),
///     Ok(false),  // false indicates the original file was updated
/// ));
///
/// flac.rewind().unwrap();
///
/// // re-read the metadata blocks from the original file
/// let blocks = BlockList::read(flac).unwrap();
///
/// // the original file now has a Vorbis Comment block
/// // with the track title that we added
/// assert_eq!(
///     blocks.get::<VorbisComment>().and_then(|vc| vc.get(TITLE)),
///     Some("Track Title"),
/// );
///
/// // the original file's padding block is smaller than before
/// // to accomodate our new Vorbis Comment block
/// assert!(u32::from(blocks.get::<Padding>().unwrap().size) < PADDING_SIZE);
///
/// // and the unneeded rebuilt file remains empty
/// assert!(rebuilt.is_empty());
/// ```
///
/// # Example 2
///
/// ```
/// use flac_codec::{
///     metadata::{update_file, BlockList, VorbisComment},
///     metadata::fields::TITLE,
///     encode::{FlacSampleWriter, Options},
/// };
/// use std::io::{Cursor, Seek};
///
/// let mut flac = Cursor::new(vec![]);  // a FLAC file in memory
///
/// // include no padding in our encoded file
/// let options = Options::default().no_padding();
///
/// let mut writer = FlacSampleWriter::new(
///     &mut flac,  // our wrapped writer
///     options,    // our encoding options
///     44100,      // sample rate
///     16,         // bits-per-sample
///     1,          // channel count
///     Some(1),    // total samples
/// ).unwrap();
///
/// // write a simple FLAC file
/// writer.write(std::slice::from_ref(&0)).unwrap();
/// writer.finalize().unwrap();
///
/// flac.rewind().unwrap();
///
/// let mut rebuilt: Vec<u8> = vec![];
///
/// // update file with new Vorbis Comment
/// assert!(matches!(
///     update_file::<_, _, flac_codec::Error>(
///         // our original file
///         &mut flac,
///         // a closure to create a new file, if necessary
///         || Ok(&mut rebuilt),
///         // the closure that performs the metadata update
///         |blocklist| {
///             blocklist.update::<VorbisComment>(
///                 // the blocklist itself has a closure
///                 // that updates a block, creating it if necessary
///                 // (in this case, we're updating the Vorbis comment)
///                 |vc| vc.set(TITLE, "Track Title")
///             );
///             Ok(())
///         },
///     ),
///     Ok(true),  // true indicates the original file was not updated
/// ));
///
/// flac.rewind().unwrap();
///
/// // re-read the metadata blocks from the original file
/// let blocks = BlockList::read(flac).unwrap();
///
/// // the original file remains unchanged
/// // and has no Vorbis Comment block
/// assert_eq!(blocks.get::<VorbisComment>(), None);
///
/// // now read the metadata blocks from the rebuilt file
/// let blocks = BlockList::read(rebuilt.as_slice()).unwrap();
///
/// // the rebuilt file has our Vorbis Comment entry instead
/// assert_eq!(
///     blocks.get::<VorbisComment>().and_then(|vc| vc.get(TITLE)),
///     Some("Track Title"),
/// );
/// ```
pub fn update_file<F, N, E>(
    mut original: F,
    rebuilt: impl FnOnce() -> std::io::Result<N>,
    f: impl FnOnce(&mut BlockList) -> Result<(), E>,
) -> Result<bool, E>
where
    F: std::io::Read + std::io::Seek + std::io::Write,
    N: std::io::Write,
    E: From<Error>,
{
    use crate::Counter;
    use std::cmp::Ordering;
    use std::io::{BufReader, BufWriter, Read, sink};

    fn rebuild_file<N, R>(
        rebuilt: impl FnOnce() -> std::io::Result<N>,
        mut r: R,
        blocks: BlockList,
    ) -> Result<(), Error>
    where
        N: std::io::Write,
        R: Read,
    {
        // dump our new blocks and remaining FLAC data to temp file
        let mut tmp = Vec::new();
        write_blocks(&mut tmp, blocks)?;
        std::io::copy(&mut r, &mut tmp).map_err(Error::Io)?;
        drop(r);

        // fresh original file and rewrite it with tmp file contents
        rebuilt()
            .and_then(|mut f| f.write_all(tmp.as_slice()))
            .map_err(Error::Io)
    }

    /// Returns Ok if successful
    fn grow_padding(blocks: &mut BlockList, more_bytes: u64) -> Result<(), ()> {
        // if a block set has more than one PADDING, we'll try the first
        // rather than attempt to grow each in turn
        //
        // this is the most common case
        let padding = blocks.get_mut::<Padding>().ok_or(())?;

        padding.size = padding
            .size
            .checked_add(more_bytes.try_into().map_err(|_| ())?)
            .ok_or(())?;

        Ok(())
    }

    /// Returns Ok if successful
    fn shrink_padding(blocks: &mut BlockList, fewer_bytes: u64) -> Result<(), ()> {
        // if a block set has more than one PADDING, we'll try the first
        // rather than attempt to grow each in turn
        //
        // this is the most common case
        let padding = blocks.get_mut::<Padding>().ok_or(())?;

        padding.size = padding
            .size
            .checked_sub(fewer_bytes.try_into().map_err(|_| ())?)
            .ok_or(())?;

        Ok(())
    }

    // the starting position in the stream we rewind to
    let start = std::io::SeekFrom::Start(original.stream_position().map_err(Error::Io)?);

    let mut reader = Counter::new(BufReader::new(&mut original));

    let mut blocks = BlockList::read(Read::by_ref(&mut reader))?;

    let Counter {
        stream: reader,
        count: old_size,
    } = reader;

    f(&mut blocks)?;

    let new_size = {
        let mut new_size = Counter::new(sink());
        write_blocks(&mut new_size, blocks.blocks())?;
        new_size.count
    };

    match new_size.cmp(&old_size) {
        Ordering::Less => {
            // blocks have shrunk in size, so try to expand
            // PADDING block to hold additional bytes
            match grow_padding(&mut blocks, old_size - new_size) {
                Ok(()) => {
                    original.seek(start).map_err(Error::Io)?;
                    write_blocks(BufWriter::new(original), blocks)
                        .map(|()| false)
                        .map_err(E::from)
                }
                Err(()) => rebuild_file(rebuilt, reader, blocks)
                    .map(|()| true)
                    .map_err(E::from),
            }
        }
        Ordering::Equal => {
            // blocks are the same size, so no need to adjust padding
            original.seek(start).map_err(Error::Io)?;
            write_blocks(BufWriter::new(original), blocks)
                .map(|()| false)
                .map_err(E::from)
        }
        Ordering::Greater => {
            // blocks have grown in size, so try to shrink
            // PADDING block to hold additional bytes
            match shrink_padding(&mut blocks, new_size - old_size) {
                Ok(()) => {
                    original.seek(start).map_err(Error::Io)?;
                    write_blocks(BufWriter::new(original), blocks)
                        .map(|()| false)
                        .map_err(E::from)
                }
                Err(()) => rebuild_file(rebuilt, reader, blocks)
                    .map(|()| true)
                    .map_err(E::from),
            }
        }
    }
}

/// Any possible FLAC metadata block
///
/// Each block consists of a [`BlockHeader`] followed by the block's contents.
///
/// ```text
/// ┌──────────┬────────┬┄┄┄┄┄┄┄┄┬┄┄┄┬────────┬┄┄┄┄┄┄┄┄┬┄┄┄╮
/// │ FLAC Tag │ Block₀ │ Block₁ ┆ … ┆ Frame₀ │ Frame₁ ┆ … ┆ FLAC File
/// └──────────┼────────┼┄┄┄┄┄┄┄┄┴┄┄┄┴────────┴┄┄┄┄┄┄┄┄┴┄┄┄╯
/// ╭──────────╯        ╰────────────────────────╮
/// ├──────────────┬─────────────────────────────┤
/// │ Block Header │     Metadata Block Data     │           Metadata Block
/// └──────────────┴─────────────────────────────┘
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Block {
    /// The STREAMINFO block
    Streaminfo(Streaminfo),
    /// The PADDING block
    Padding(Padding),
    /// The APPLICATION block
    Application(Application),
    /// The SEEKTABLE block
    SeekTable(SeekTable),
    /// The VORBIS_COMMENT block
    VorbisComment(VorbisComment),
    /// The CUESHEET block
    Cuesheet(Cuesheet),
    /// The PICTURE block
    Picture(Picture),
}

impl Block {
    /// Our block type
    pub fn block_type(&self) -> BlockType {
        match self {
            Self::Streaminfo(_) => BlockType::Streaminfo,
            Self::Padding(_) => BlockType::Padding,
            Self::Application(_) => BlockType::Application,
            Self::SeekTable(_) => BlockType::SeekTable,
            Self::VorbisComment(_) => BlockType::VorbisComment,
            Self::Cuesheet(_) => BlockType::Cuesheet,
            Self::Picture(_) => BlockType::Picture,
        }
    }
}

impl AsBlockRef for Block {
    fn as_block_ref(&self) -> BlockRef<'_> {
        match self {
            Self::Streaminfo(s) => BlockRef::Streaminfo(s),
            Self::Padding(p) => BlockRef::Padding(p),
            Self::Application(a) => BlockRef::Application(a),
            Self::SeekTable(s) => BlockRef::SeekTable(s),
            Self::VorbisComment(v) => BlockRef::VorbisComment(v),
            Self::Cuesheet(v) => BlockRef::Cuesheet(v),
            Self::Picture(p) => BlockRef::Picture(p),
        }
    }
}

impl FromBitStreamWith<'_> for Block {
    type Context = BlockHeader;
    type Error = Error;

    // parses from reader without header
    fn from_reader<R: BitRead + ?Sized>(
        r: &mut R,
        header: &BlockHeader,
    ) -> Result<Self, Self::Error> {
        match header.block_type {
            BlockType::Streaminfo => Ok(Block::Streaminfo(r.parse()?)),
            BlockType::Padding => Ok(Block::Padding(r.parse_using(header.size)?)),
            BlockType::Application => Ok(Block::Application(r.parse_using(header.size)?)),
            BlockType::SeekTable => Ok(Block::SeekTable(r.parse_using(header.size)?)),
            BlockType::VorbisComment => Ok(Block::VorbisComment(r.parse()?)),
            BlockType::Cuesheet => Ok(Block::Cuesheet(r.parse()?)),
            BlockType::Picture => Ok(Block::Picture(r.parse()?)),
        }
    }
}

impl ToBitStreamUsing for Block {
    type Context = bool;
    type Error = Error;

    // builds to writer with header
    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W, is_last: bool) -> Result<(), Error> {
        match self {
            Self::Streaminfo(streaminfo) => w
                .build(&BlockHeader::new(is_last, streaminfo)?)
                .and_then(|()| w.build(streaminfo).map_err(Error::Io)),
            Self::Padding(padding) => w
                .build(&BlockHeader::new(is_last, padding)?)
                .and_then(|()| w.build(padding).map_err(Error::Io)),
            Self::Application(application) => w
                .build(&BlockHeader::new(is_last, application)?)
                .and_then(|()| w.build(application).map_err(Error::Io)),
            Self::SeekTable(seektable) => w
                .build(&BlockHeader::new(is_last, seektable)?)
                .and_then(|()| w.build(seektable)),
            Self::VorbisComment(vorbis_comment) => w
                .build(&BlockHeader::new(is_last, vorbis_comment)?)
                .and_then(|()| w.build(vorbis_comment)),
            Self::Cuesheet(cuesheet) => w
                .build(&BlockHeader::new(is_last, cuesheet)?)
                .and_then(|()| w.build(cuesheet)),
            Self::Picture(picture) => w
                .build(&BlockHeader::new(is_last, picture)?)
                .and_then(|()| w.build(picture)),
        }
    }
}

/// A shared reference to a metadata block
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BlockRef<'b> {
    /// The STREAMINFO block
    Streaminfo(&'b Streaminfo),
    /// The PADDING block
    Padding(&'b Padding),
    /// The APPLICATION block
    Application(&'b Application),
    /// The SEEKTABLE block
    SeekTable(&'b SeekTable),
    /// The VORBIS_COMMENT block
    VorbisComment(&'b VorbisComment),
    /// The CUESHEET block
    Cuesheet(&'b Cuesheet),
    /// The PICTURE block
    Picture(&'b Picture),
}

impl BlockRef<'_> {
    /// Our block type
    pub fn block_type(&self) -> BlockType {
        match self {
            Self::Streaminfo(_) => BlockType::Streaminfo,
            Self::Padding(_) => BlockType::Padding,
            Self::Application(_) => BlockType::Application,
            Self::SeekTable(_) => BlockType::SeekTable,
            Self::VorbisComment(_) => BlockType::VorbisComment,
            Self::Cuesheet(_) => BlockType::Cuesheet,
            Self::Picture(_) => BlockType::Picture,
        }
    }
}

impl AsBlockRef for BlockRef<'_> {
    fn as_block_ref(&self) -> BlockRef<'_> {
        *self
    }
}

/// A trait for items which can make cheap [`BlockRef`] values.
pub trait AsBlockRef {
    /// Returns fresh reference to ourself.
    fn as_block_ref(&self) -> BlockRef<'_>;
}

impl<T: AsBlockRef> AsBlockRef for &T {
    fn as_block_ref(&self) -> BlockRef<'_> {
        <T as AsBlockRef>::as_block_ref(*self)
    }
}

impl ToBitStreamUsing for BlockRef<'_> {
    type Context = bool;
    type Error = Error;

    // builds to writer with header
    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W, is_last: bool) -> Result<(), Error> {
        match self {
            Self::Streaminfo(streaminfo) => w
                .build(&BlockHeader::new(is_last, *streaminfo)?)
                .and_then(|()| w.build(*streaminfo).map_err(Error::Io)),
            Self::Padding(padding) => w
                .build(&BlockHeader::new(is_last, *padding)?)
                .and_then(|()| w.build(*padding).map_err(Error::Io)),
            Self::Application(application) => w
                .build(&BlockHeader::new(is_last, *application)?)
                .and_then(|()| w.build(*application).map_err(Error::Io)),
            Self::SeekTable(seektable) => w
                .build(&BlockHeader::new(is_last, *seektable)?)
                .and_then(|()| w.build(*seektable)),
            Self::VorbisComment(vorbis_comment) => w
                .build(&BlockHeader::new(is_last, *vorbis_comment)?)
                .and_then(|()| w.build(*vorbis_comment)),
            Self::Cuesheet(cuesheet) => w
                .build(&BlockHeader::new(is_last, *cuesheet)?)
                .and_then(|()| w.build(*cuesheet)),
            Self::Picture(picture) => w
                .build(&BlockHeader::new(is_last, *picture)?)
                .and_then(|()| w.build(*picture)),
        }
    }
}

macro_rules! block {
    ($t:ty, $v:ident, $m:literal) => {
        impl MetadataBlock for $t {
            const TYPE: BlockType = BlockType::$v;
            const MULTIPLE: bool = $m;
        }

        impl From<$t> for Block {
            fn from(b: $t) -> Self {
                Self::$v(b)
            }
        }

        impl TryFrom<Block> for $t {
            type Error = ();

            fn try_from(block: Block) -> Result<Self, ()> {
                match block {
                    Block::$v(block) => Ok(block),
                    _ => Err(()),
                }
            }
        }

        impl AsBlockRef for $t {
            fn as_block_ref(&self) -> BlockRef<'_> {
                BlockRef::$v(self)
            }
        }
    };
}

macro_rules! optional_block {
    ($t:ty, $v:ident) => {
        impl OptionalMetadataBlock for $t {
            const OPTIONAL_TYPE: OptionalBlockType = OptionalBlockType::$v;
        }

        impl private::OptionalMetadataBlock for $t {
            fn try_from_opt_block(
                block: &private::OptionalBlock,
            ) -> Result<&Self, &private::OptionalBlock> {
                match block {
                    private::OptionalBlock::$v(block) => Ok(block),
                    block => Err(block),
                }
            }

            fn try_from_opt_block_mut(
                block: &mut private::OptionalBlock,
            ) -> Result<&mut Self, &mut private::OptionalBlock> {
                match block {
                    private::OptionalBlock::$v(block) => Ok(block),
                    block => Err(block),
                }
            }
        }

        impl From<$t> for private::OptionalBlock {
            fn from(vorbis: $t) -> Self {
                private::OptionalBlock::$v(vorbis)
            }
        }

        impl TryFrom<private::OptionalBlock> for $t {
            type Error = ();

            fn try_from(block: private::OptionalBlock) -> Result<Self, ()> {
                match block {
                    private::OptionalBlock::$v(b) => Ok(b),
                    _ => Err(()),
                }
            }
        }
    };
}

/// A STREAMINFO metadata block
///
/// This block contains metadata about the stream's contents.
///
/// It must *always* be present in a FLAC file,
/// must *always* be the first metadata block in the stream,
/// and must *not* be present more than once.
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 16   | `minimum_block_size` | minimum block size (in samples) in the stream
/// | 16   | `maximum_block_size` | maximum block size (in samples) in the stream
/// | 24   | `minimum_frame_size` | minimum frame size (in bytes) in the stream
/// | 24   | `maximum_frame_size` | maximum frame size (in bytes) in the stream
/// | 20   | `sample_rate` | stream's sample rate, in Hz
/// | 3    | `channels` | stream's channel count (+1)
/// | 5    | `bits_per_sample` | stream's bits-per-sample (+1)
/// | 36   | `total_samples` | stream's total channel-independent samples
/// | 16×8 | `md5` | decoded stream's MD5 sum hash
///
/// # Example
/// ```
/// use bitstream_io::{BitReader, BitRead, BigEndian, SignedBitCount};
/// use flac_codec::metadata::Streaminfo;
/// use std::num::NonZero;
///
/// let data: &[u8] = &[
///     0x10, 0x00,
///     0x10, 0x00,
///     0x00, 0x00, 0x0c,
///     0x00, 0x00, 0x0c,
///     0b00001010, 0b11000100, 0b0100_000_0, 0b1111_0000,
///     0b00000000, 0b00000000, 0b00000000, 0b01010000,
///     0xf5, 0x3f, 0x86, 0x87, 0x6d, 0xcd, 0x77, 0x83,
///     0x22, 0x5c, 0x93, 0xba, 0x8a, 0x93, 0x8c, 0x7d,
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// assert_eq!(
///     r.parse::<Streaminfo>().unwrap(),
///     Streaminfo {
///         minimum_block_size: 0x10_00,                    // 4096 samples
///         maximum_block_size: 0x10_00,                    // 4096 samples
///         minimum_frame_size: NonZero::new(0x00_00_0c),   // 12 bytes
///         maximum_frame_size: NonZero::new(0x00_00_0c),   // 12 bytes
///         sample_rate: 0b00001010_11000100_0100,          // 44100 Hz
///         channels: NonZero::new(0b000 + 1).unwrap(),     // 1 channel
///         bits_per_sample:
///             SignedBitCount::new::<{0b0_1111 + 1}>(),    // 16 bps
///         total_samples: NonZero::new(
///             0b0000_00000000_00000000_00000000_01010000  // 80 samples
///         ),
///         md5: Some([
///             0xf5, 0x3f, 0x86, 0x87, 0x6d, 0xcd, 0x77, 0x83,
///             0x22, 0x5c, 0x93, 0xba, 0x8a, 0x93, 0x8c, 0x7d,
///         ]),
///     },
/// );
/// ```
///
/// # Important
///
/// Changing any of these values to something that differs
/// from the values of the file's frame headers will render it
/// unplayable, as will moving it anywhere but the first
/// metadata block in the file.
/// Avoid modifying the position and contents of this block unless you
/// know exactly what you are doing.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Streaminfo {
    /// The minimum block size (in samples) used in the stream,
    /// excluding the last block.
    pub minimum_block_size: u16,
    /// The maximum block size (in samples) used in the stream,
    /// excluding the last block.
    pub maximum_block_size: u16,
    /// The minimum framesize (in bytes) used in the stream.
    ///
    /// `None` indicates the value is unknown.
    pub minimum_frame_size: Option<NonZero<u32>>,
    /// The maximum framesize (in bytes) used in the stream.
    ///
    /// `None` indicates the value is unknown.
    pub maximum_frame_size: Option<NonZero<u32>>,
    /// Sample rate in Hz
    ///
    /// 0 indicates a non-audio stream.
    pub sample_rate: u32,
    /// Number of channels, from 1 to 8
    pub channels: NonZero<u8>,
    /// Number of bits-per-sample, from 4 to 32
    pub bits_per_sample: SignedBitCount<32>,
    /// Total number of interchannel samples in stream.
    ///
    /// `None` indicates the value is unknown.
    pub total_samples: Option<NonZero<u64>>,
    /// MD5 hash of unencoded audio data.
    ///
    /// `None` indicates the value is unknown.
    pub md5: Option<[u8; 16]>,
}

impl Streaminfo {
    /// The maximum size of a frame, in bytes (2²⁴ - 1)
    pub const MAX_FRAME_SIZE: u32 = (1 << 24) - 1;

    /// The maximum sample rate, in Hz (2²⁰ - 1)
    pub const MAX_SAMPLE_RATE: u32 = (1 << 20) - 1;

    /// The maximum number of channels (8)
    pub const MAX_CHANNELS: NonZero<u8> = NonZero::new(8).unwrap();

    /// The maximum number of total samples (2³⁶ - 1)
    pub const MAX_TOTAL_SAMPLES: NonZero<u64> = NonZero::new((1 << 36) - 1).unwrap();

    /// Defined size of STREAMINFO block
    const SIZE: BlockSize = BlockSize(0x22);
}

block!(Streaminfo, Streaminfo, false);

impl Metadata for Streaminfo {
    fn channel_count(&self) -> u8 {
        self.channels.get()
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample.into()
    }

    fn total_samples(&self) -> Option<u64> {
        self.total_samples.map(|s| s.get())
    }

    fn md5(&self) -> Option<&[u8; 16]> {
        self.md5.as_ref()
    }
}

impl FromBitStream for Streaminfo {
    type Error = std::io::Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        Ok(Self {
            minimum_block_size: r.read_to()?,
            maximum_block_size: r.read_to()?,
            minimum_frame_size: r.read::<24, _>()?,
            maximum_frame_size: r.read::<24, _>()?,
            sample_rate: r.read::<20, _>()?,
            channels: r.read::<3, _>()?,
            bits_per_sample: r
                .read_count::<0b11111>()?
                .checked_add(1)
                .and_then(|c| c.signed_count())
                .unwrap(),
            total_samples: r.read::<36, _>()?,
            md5: r
                .read_to()
                .map(|md5: [u8; 16]| md5.iter().any(|b| *b != 0).then_some(md5))?,
        })
    }
}

impl ToBitStream for Streaminfo {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(self.minimum_block_size)?;
        w.write_from(self.maximum_block_size)?;
        w.write::<24, _>(self.minimum_frame_size)?;
        w.write::<24, _>(self.maximum_frame_size)?;
        w.write::<20, _>(self.sample_rate)?;
        w.write::<3, _>(self.channels)?;
        w.write_count(
            self.bits_per_sample
                .checked_sub::<0b11111>(1)
                .unwrap()
                .count(),
        )?;
        w.write::<36, _>(self.total_samples)?;
        w.write_from(self.md5.unwrap_or([0; 16]))?;
        Ok(())
    }
}

/// A PADDING metadata block
///
/// Padding blocks are empty blocks consisting of all 0 bytes.
/// If one wishes to edit the metadata in other blocks,
/// adjusting the size of the padding block allows
/// us to do so without have to rewrite the entire FLAC file.
/// For example, when adding 10 bytes to a comment,
/// we can subtract 10 bytes from the padding
/// and the total size of all blocks remains unchanged.
/// Therefore we can simply overwrite the old comment
/// block with the new without affecting the following
/// FLAC audio frames.
///
/// This block may occur multiple times in a FLAC file.
///
/// # Example
///
/// ```
/// use bitstream_io::{BitReader, BitRead, BigEndian};
/// use flac_codec::metadata::{BlockHeader, BlockType, Padding};
///
/// let data: &[u8] = &[
///     0x81, 0x00, 0x00, 0x0a,  // block header
///     // padding bytes
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// let header = r.parse::<BlockHeader>().unwrap();
/// assert_eq!(
///     &header,
///     &BlockHeader {
///         last: true,
///         block_type: BlockType::Padding,
///         size: 0x0au8.into(),
///     },
/// );
///
/// assert_eq!(
///     r.parse_using::<Padding>(header.size).unwrap(),
///     Padding {
///         size: 0x0au8.into(),
///     },
/// );
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct Padding {
    /// The size of the padding, in bytes
    pub size: BlockSize,
}

block!(Padding, Padding, true);
optional_block!(Padding, Padding);

impl FromBitStreamUsing for Padding {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: BlockSize) -> Result<Self, Self::Error> {
        r.skip(size.get() * 8)?;
        Ok(Self { size })
    }
}

impl ToBitStream for Padding {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.pad(self.size.get() * 8)
    }
}

/// An APPLICATION metadata block
///
/// This block is for handling application-specific binary metadata,
/// such as foreign RIFF WAVE tags.
///
/// This block may occur multiple times in a FLAC file.
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 32   | `id` | registered application ID
/// | rest of block | `data` | application-specific data
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Application {
    /// A registered application ID
    pub id: u32,
    /// Application-specific data
    pub data: Vec<u8>,
}

impl Application {
    /// Application ID for RIFF chunk storage
    pub const RIFF: u32 = 0x72696666;

    /// Application ID for AIFF chunk storage
    pub const AIFF: u32 = 0x61696666;
}

block!(Application, Application, true);
optional_block!(Application, Application);

impl FromBitStreamUsing for Application {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: BlockSize) -> Result<Self, Self::Error> {
        Ok(Self {
            id: r.read_to()?,
            data: r.read_to_vec(
                size.get()
                    .checked_sub(4)
                    .ok_or(Error::InsufficientApplicationBlock)?
                    .try_into()
                    .unwrap(),
            )?,
        })
    }
}

impl ToBitStream for Application {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(self.id)?;
        w.write_bytes(&self.data)
    }
}

/// A SEEKTABLE metadata block
///
/// Because FLAC frames do not store their compressed length,
/// a seek table is used for random access within a FLAC file.
/// By mapping a sample number to a byte offset,
/// one can quickly reach different parts of the file
/// without decoding the whole thing.
///
/// Also note that seek point byte offsets are
/// relative to the start of the first FLAC frame,
/// and *not* relative to the start of the entire file.
/// This allows us to change the size of the set
/// of metadata blocks without having to recalculate
/// the contents of the seek table.
///
/// Because the byte and sample offsets are
/// file-specific, a seek table generated for one file
/// should not be transferred to another FLAC file where the
/// frames are different sizes and in different positions.
///
/// This block may occur only once in a FLAC file.
///
/// Its seekpoints occupy the entire block.
///
/// # Example
/// ```
/// use bitstream_io::{BitReader, BitRead, BigEndian};
/// use flac_codec::metadata::{BlockHeader, BlockType, SeekTable, SeekPoint};
///
/// let data: &[u8] = &[
///     0x83, 0x00, 0x00, 0x48,  // block header
///     // seekpoint 0
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
///     0x00, 0x14,
///     // seekpoint 1
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14,
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c,
///     0x00, 0x14,
///     // seekpoint 2
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28,
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x22,
///     0x00, 0x14,
///     // seekpoint 3
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c,
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c,
///     0x00, 0x14,
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// let header = r.parse::<BlockHeader>().unwrap();
/// assert_eq!(
///     &header,
///     &BlockHeader {
///         last: true,
///         block_type: BlockType::SeekTable,
///         size: 0x48u8.into(),
///     },
/// );
///
/// let seektable = r.parse_using::<SeekTable>(header.size).unwrap();
///
/// assert_eq!(
///     Vec::from(seektable.points),
///     vec![
///         SeekPoint::Defined {
///             sample_offset: 0x00,
///             byte_offset: 0x00,
///             frame_samples: 0x14,
///         },
///         SeekPoint::Defined {
///             sample_offset: 0x14,
///             byte_offset: 0x0c,
///             frame_samples: 0x14,
///         },
///         SeekPoint::Defined {
///             sample_offset: 0x28,
///             byte_offset: 0x22,
///             frame_samples: 0x14,
///         },
///         SeekPoint::Defined {
///             sample_offset: 0x3c,
///             byte_offset: 0x3c,
///             frame_samples: 0x14,
///         },
///     ],
/// );
///
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct SeekTable {
    /// The seek table's individual seek points
    pub points: contiguous::Contiguous<{ Self::MAX_POINTS }, SeekPoint>,
}

impl SeekTable {
    /// The maximum number of seek points that fit into a seek table
    pub const MAX_POINTS: usize = (1 << 24) / ((64 + 64 + 16) / 8);
}

block!(SeekTable, SeekTable, false);
optional_block!(SeekTable, SeekTable);

impl FromBitStreamUsing for SeekTable {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: BlockSize) -> Result<Self, Self::Error> {
        match (size.get() / 18, size.get() % 18) {
            (p, 0) => Ok(Self {
                points: contiguous::Contiguous::try_collect((0..p).map(|_| r.parse()))
                    .map_err(|_| Error::InvalidSeekTablePoint)??,
            }),
            _ => Err(Error::InvalidSeekTableSize),
        }
    }
}

impl ToBitStream for SeekTable {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        // ensure non-placeholder seek point offsets increment
        let mut last_offset = None;

        self.points
            .iter()
            .try_for_each(|point| match last_offset.as_mut() {
                None => {
                    last_offset = point.sample_offset();
                    w.build(point).map_err(Error::Io)
                }
                Some(last_offset) => match point.sample_offset() {
                    Some(our_offset) => match our_offset > *last_offset {
                        true => {
                            *last_offset = our_offset;
                            w.build(point).map_err(Error::Io)
                        }
                        false => Err(Error::InvalidSeekTablePoint),
                    },
                    _ => w.build(point).map_err(Error::Io),
                },
            })
    }
}

/// An individual SEEKTABLE seek point
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 64   | `sample_offset` | sample number of first sample in target frame
/// | 64   | `byte_offset` | offset, in bytes, from first frame to target frame's header
/// | 16   | `frame_samples` | number of samples in target frame
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SeekPoint {
    /// A defined, non-placeholder seek point
    Defined {
        /// The sample number of the first sample in the target frame
        sample_offset: u64,
        /// Offset, in bytes, from the first byte of the first frame header
        /// to the first byte in the target frame's header
        byte_offset: u64,
        /// Number of samples in the target frame
        frame_samples: u16,
    },
    /// A placeholder seek point
    Placeholder,
}

impl SeekPoint {
    /// Returns our sample offset, if not a placeholder point
    pub fn sample_offset(&self) -> Option<u64> {
        match self {
            Self::Defined { sample_offset, .. } => Some(*sample_offset),
            Self::Placeholder => None,
        }
    }
}

impl contiguous::Adjacent for SeekPoint {
    fn valid_first(&self) -> bool {
        true
    }

    fn is_next(&self, previous: &SeekPoint) -> bool {
        // seekpoints must be unique by sample offset
        // and sample offsets must be ascending
        //
        // placeholders can come after non-placeholders or other placeholders,
        // but non-placeholders can't come after placeholders
        match self {
            Self::Defined {
                sample_offset: our_offset,
                ..
            } => match previous {
                Self::Defined {
                    sample_offset: prev_offset,
                    ..
                } => our_offset > prev_offset,
                Self::Placeholder => false,
            },
            Self::Placeholder => true,
        }
    }
}

impl FromBitStream for SeekPoint {
    type Error = std::io::Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        match r.read_to()? {
            u64::MAX => {
                let _byte_offset = r.read_to::<u64>()?;
                let _frame_samples = r.read_to::<u16>()?;
                Ok(Self::Placeholder)
            }
            sample_offset => Ok(Self::Defined {
                sample_offset,
                byte_offset: r.read_to()?,
                frame_samples: r.read_to()?,
            }),
        }
    }
}

impl ToBitStream for SeekPoint {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        match self {
            Self::Defined {
                sample_offset,
                byte_offset,
                frame_samples,
            } => {
                w.write_from(*sample_offset)?;
                w.write_from(*byte_offset)?;
                w.write_from(*frame_samples)
            }
            Self::Placeholder => {
                w.write_from(u64::MAX)?;
                w.write_from::<u64>(0)?;
                w.write_from::<u16>(0)
            }
        }
    }
}

/// A VORBIS_COMMENT metadata block
///
/// This block contains metadata such as track name,
/// artist name, album name, etc.  Its contents are
/// UTF-8 encoded, `=`-delimited text fields
/// with a field name followed by value,
/// such as:
///
/// ```text
/// TITLE=Track Title
/// ```
///
/// Field names are case-insensitive and
/// may occur multiple times within the same comment
/// (a track may have multiple artists and choose to
/// store an "ARTIST" field for each one).
///
/// Commonly-used fields are available in the [`fields`] module.
///
/// This block may occur only once in a FLAC file.
///
/// # Byte Order
///
/// Unlike the rest of a FLAC file, the Vorbis comment's
/// length fields are stored in little-endian byte order.
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 32   | vendor string len | length of vendor string, in bytes
/// | `vendor string len`×8 | `vendor_string` | vendor string, in UTF-8
/// | 32   | field count | number of vendor string fields
/// | 32   | field₀ len | length of field₀, in bytes
/// | `field₀ len`×8 | `fields₀` | first field value, in UTF-8
/// | 32   | field₁ len | length of field₁, in bytes
/// | `field₁ len`×8 | `fields₁` | second field value, in UTF-8
/// | | | ⋮
///
/// # Example
/// ```
/// use bitstream_io::{BitReader, BitRead, LittleEndian};
/// use flac_codec::metadata::VorbisComment;
/// use flac_codec::metadata::fields::{TITLE, ALBUM, ARTIST};
///
/// let data: &[u8] = &[
///     0x20, 0x00, 0x00, 0x00,  // 32 byte vendor string
///     0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63,
///     0x65, 0x20, 0x6c, 0x69, 0x62, 0x46, 0x4c, 0x41,
///     0x43, 0x20, 0x31, 0x2e, 0x34, 0x2e, 0x33, 0x20,
///     0x32, 0x30, 0x32, 0x33, 0x30, 0x36, 0x32, 0x33,
///     0x02, 0x00, 0x00, 0x00,  // 2 fields
///     0x0d, 0x00, 0x00, 0x00,  // 13 byte field 1
///     0x54, 0x49, 0x54, 0x4c, 0x45, 0x3d, 0x54, 0x65,
///     0x73, 0x74, 0x69, 0x6e, 0x67,
///     0x10, 0x00, 0x00, 0x00,  // 16 byte field 2
///     0x41, 0x4c, 0x42, 0x55, 0x4d, 0x3d, 0x54, 0x65,
///     0x73, 0x74, 0x20, 0x41, 0x6c, 0x62, 0x75, 0x6d,
/// ];
///
/// let mut r = BitReader::endian(data, LittleEndian);
/// let comment = r.parse::<VorbisComment>().unwrap();
///
/// assert_eq!(
///     &comment,
///     &VorbisComment {
///         vendor_string: "reference libFLAC 1.4.3 20230623".to_string(),
///         fields: vec![
///              "TITLE=Testing".to_string(),
///              "ALBUM=Test Album".to_string(),
///         ],
///     },
/// );
///
/// assert_eq!(comment.get(TITLE), Some("Testing"));
/// assert_eq!(comment.get(ALBUM), Some("Test Album"));
/// assert_eq!(comment.get(ARTIST), None);
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct VorbisComment {
    /// The vendor string
    pub vendor_string: String,
    /// The individual metadata comment strings
    pub fields: Vec<String>,
}

impl Default for VorbisComment {
    fn default() -> Self {
        Self {
            vendor_string: concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"))
                .to_owned(),
            fields: vec![],
        }
    }
}

impl VorbisComment {
    /// Given a field name, returns first matching value, if any
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{VorbisComment, fields::{ARTIST, TITLE}};
    ///
    /// let comment = VorbisComment {
    ///     fields: vec![
    ///         "ARTIST=Artist 1".to_owned(),
    ///         "ARTIST=Artist 2".to_owned(),
    ///     ],
    ///     ..VorbisComment::default()
    /// };
    ///
    /// assert_eq!(comment.get(ARTIST), Some("Artist 1"));
    /// assert_eq!(comment.get(TITLE), None);
    /// ```
    pub fn get(&self, field: &str) -> Option<&str> {
        self.all(field).next()
    }

    /// Replaces any instances of the given field with value
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Panics
    ///
    /// Panics if field contains the `=` character.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{VorbisComment, fields::ARTIST};
    ///
    /// let mut comment = VorbisComment {
    ///     fields: vec![
    ///         "ARTIST=Artist 1".to_owned(),
    ///         "ARTIST=Artist 2".to_owned(),
    ///     ],
    ///     ..VorbisComment::default()
    /// };
    ///
    /// comment.set(ARTIST, "Artist 3");
    ///
    /// assert_eq!(
    ///     comment.all(ARTIST).collect::<Vec<_>>(),
    ///     vec!["Artist 3"],
    /// );
    /// ```
    pub fn set<S>(&mut self, field: &str, value: S)
    where
        S: std::fmt::Display,
    {
        self.remove(field);
        self.insert(field, value);
    }

    /// Given a field name, iterates over any matching values
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{VorbisComment, fields::ARTIST};
    ///
    /// let comment = VorbisComment {
    ///     fields: vec![
    ///         "ARTIST=Artist 1".to_owned(),
    ///         "ARTIST=Artist 2".to_owned(),
    ///     ],
    ///     ..VorbisComment::default()
    /// };
    ///
    /// assert_eq!(
    ///     comment.all(ARTIST).collect::<Vec<_>>(),
    ///     vec!["Artist 1", "Artist 2"],
    /// );
    /// ```
    pub fn all(&self, field: &str) -> impl Iterator<Item = &str> {
        assert!(!field.contains('='), "field must not contain '='");

        self.fields.iter().filter_map(|f| {
            f.split_once('=')
                .and_then(|(key, value)| key.eq_ignore_ascii_case(field).then_some(value))
        })
    }

    /// Adds new instance of field with the given value
    ///
    /// # Panics
    ///
    /// Panics if field contains the `=` character.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{VorbisComment, fields::ARTIST};
    ///
    /// let mut comment = VorbisComment {
    ///     fields: vec![
    ///         "ARTIST=Artist 1".to_owned(),
    ///         "ARTIST=Artist 2".to_owned(),
    ///     ],
    ///     ..VorbisComment::default()
    /// };
    ///
    /// comment.insert(ARTIST, "Artist 3");
    ///
    /// assert_eq!(
    ///     comment.all(ARTIST).collect::<Vec<_>>(),
    ///     vec!["Artist 1", "Artist 2", "Artist 3"],
    /// );
    /// ```
    pub fn insert<S>(&mut self, field: &str, value: S)
    where
        S: std::fmt::Display,
    {
        assert!(!field.contains('='), "field must not contain '='");

        self.fields.push(format!("{field}={value}"));
    }

    /// Removes any matching instances of the given field
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Panics
    ///
    /// Panics if field contains the `=` character.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{VorbisComment, fields::ARTIST};
    ///
    /// let mut comment = VorbisComment {
    ///     fields: vec![
    ///         "ARTIST=Artist 1".to_owned(),
    ///         "ARTIST=Artist 2".to_owned(),
    ///     ],
    ///     ..VorbisComment::default()
    /// };
    ///
    /// comment.remove(ARTIST);
    ///
    /// assert_eq!(comment.get(ARTIST), None);
    /// ```
    pub fn remove(&mut self, field: &str) {
        assert!(!field.contains('='), "field must not contain '='");

        self.fields.retain(|f| match f.split_once('=') {
            Some((key, _)) => !key.eq_ignore_ascii_case(field),
            None => true,
        });
    }

    /// Replaces any instances of the given field with the given values
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Panics
    ///
    /// Panics if field contains the `=` character
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{VorbisComment, fields::ARTIST};
    ///
    /// let mut comment = VorbisComment {
    ///     fields: vec![
    ///         "ARTIST=Artist 1".to_owned(),
    ///         "ARTIST=Artist 2".to_owned(),
    ///     ],
    ///     ..VorbisComment::default()
    /// };
    ///
    /// comment.replace(ARTIST, ["Artist 3", "Artist 4"]);
    ///
    /// assert_eq!(
    ///     comment.all(ARTIST).collect::<Vec<_>>(),
    ///     vec!["Artist 3", "Artist 4"],
    /// );
    ///
    /// // reminder that Option also implements IntoIterator
    /// comment.replace(ARTIST, Some("Artist 5"));
    ///
    /// assert_eq!(
    ///     comment.all(ARTIST).collect::<Vec<_>>(),
    ///     vec!["Artist 5"],
    /// );
    /// ```
    pub fn replace<S: std::fmt::Display>(
        &mut self,
        field: &str,
        replacements: impl IntoIterator<Item = S>,
    ) {
        self.remove(field);
        self.fields.extend(
            replacements
                .into_iter()
                .map(|value| format!("{field}={value}")),
        );
    }
}

block!(VorbisComment, VorbisComment, false);
optional_block!(VorbisComment, VorbisComment);

impl FromBitStream for VorbisComment {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        fn read_string<R: BitRead + ?Sized>(r: &mut R) -> Result<String, Error> {
            let size = r.read_as_to::<LittleEndian, u32>()?.try_into().unwrap();
            Ok(String::from_utf8(r.read_to_vec(size)?)?)
        }

        Ok(Self {
            vendor_string: read_string(r)?,
            fields: (0..(r.read_as_to::<LittleEndian, u32>()?))
                .map(|_| read_string(r))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl ToBitStream for VorbisComment {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        fn write_string<W: BitWrite + ?Sized>(w: &mut W, s: &str) -> Result<(), Error> {
            w.write_as_from::<LittleEndian, u32>(
                s.len()
                    .try_into()
                    .map_err(|_| Error::ExcessiveStringLength)?,
            )?;
            w.write_bytes(s.as_bytes())?;
            Ok(())
        }

        write_string(w, &self.vendor_string)?;
        w.write_as_from::<LittleEndian, u32>(
            self.fields
                .len()
                .try_into()
                .map_err(|_| Error::ExcessiveVorbisEntries)?,
        )?;
        self.fields.iter().try_for_each(|s| write_string(w, s))
    }
}

// As neat as it might be implement IndexMut for VorbisComment,
// the trait simply isn't compatible.  A "&mut str" is mostly
// useless, and I can't return a partial "&mut String"
// in order to assing a new string to everything after the
// initial "FIELD=" indicator.

/// Vorbis comment metadata tag fields
///
/// Not all of these fields are officially defined in the specification,
/// but they are in common use.
pub mod fields {
    /// Name of current work
    pub const TITLE: &str = "TITLE";

    /// Name of the artist generally responsible for the current work
    pub const ARTIST: &str = "ARTIST";

    /// Name of the collection the current work belongs to
    pub const ALBUM: &str = "ALBUM";

    /// The work's original composer
    pub const COMPOSER: &str = "COMPOSER";

    /// The performance's conductor
    pub const CONDUCTOR: &str = "CONDUCTOR";

    /// The current work's performer(s)
    pub const PERFORMER: &str = "PERFORMER";

    /// The album's publisher
    pub const PUBLISHER: &str = "PUBLISHER";

    /// The album's catalog number
    pub const CATALOG: &str = "CATALOG";

    /// Release date of work
    pub const DATE: &str = "DATE";

    /// Generic comment
    pub const COMMENT: &str = "COMMENT";

    /// Track number in album
    pub const TRACK_NUMBER: &str = "TRACKNUMBER";

    /// Total tracks in album
    pub const TRACK_TOTAL: &str = "TRACKTOTAL";

    /// The channel mask of multi-channel audio streams
    pub const CHANNEL_MASK: &str = "WAVEFORMATEXTENSIBLE_CHANNEL_MASK";

    /// ReplayGain track gain
    pub const RG_TRACK_GAIN: &str = "REPLAYGAIN_TRACK_GAIN";

    /// ReplayGain album gain
    pub const RG_ALBUM_GAIN: &str = "REPLAYGAIN_ALBUM_GAIN";

    /// ReplayGain track peak
    pub const RG_TRACK_PEAK: &str = "REPLAYGAIN_TRACK_PEAK";

    /// ReplayGain album peak
    pub const RG_ALBUM_PEAK: &str = "REPLAYGAIN_ALBUM_PEAK";

    /// ReplayGain reference loudness
    pub const RG_REFERENCE_LOUDNESS: &str = "REPLAYGAIN_REFERENCE_LOUDNESS";
}

/// Types for collections which must be contiguous
///
/// Used by the SEEKTABLE and CUESHEET metadata blocks
pub mod contiguous {
    /// A trait for types which can be contiguous
    pub trait Adjacent {
        /// Whether the item is valid as the first in a sequence
        fn valid_first(&self) -> bool;

        /// Whether the item is immediately following the previous
        fn is_next(&self, previous: &Self) -> bool;
    }

    impl Adjacent for u64 {
        fn valid_first(&self) -> bool {
            *self == 0
        }

        fn is_next(&self, previous: &Self) -> bool {
            *self > *previous
        }
    }

    impl Adjacent for std::num::NonZero<u8> {
        fn valid_first(&self) -> bool {
            *self == Self::MIN
        }

        fn is_next(&self, previous: &Self) -> bool {
            previous.checked_add(1).map(|n| n == *self).unwrap_or(false)
        }
    }

    /// A Vec-like type which requires all items to be adjacent
    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct Contiguous<const MAX: usize, T: Adjacent> {
        items: Vec<T>,
    }

    impl<const MAX: usize, T: Adjacent> Default for Contiguous<MAX, T> {
        fn default() -> Self {
            Self { items: Vec::new() }
        }
    }

    impl<const MAX: usize, T: Adjacent> Contiguous<MAX, T> {
        /// Constructs new contiguous block with the given capacity
        ///
        /// See [`Vec::with_capacity`]
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                items: Vec::with_capacity(capacity.min(MAX)),
            }
        }

        /// Removes all items without adjust total capacity
        ///
        /// See [`Vec::clear`]
        pub fn clear(&mut self) {
            self.items.clear()
        }

        /// Attempts to push item into contiguous set
        ///
        /// # Errors
        ///
        /// Returns error if item is not a valid first item
        /// in the set or is not contiguous with the
        /// existing items.
        pub fn try_push(&mut self, item: T) -> Result<(), NonContiguous> {
            if self.items.len() < MAX {
                if match self.items.last() {
                    None => item.valid_first(),
                    Some(last) => item.is_next(last),
                } {
                    self.items.push(item);
                    Ok(())
                } else {
                    Err(NonContiguous)
                }
            } else {
                Err(NonContiguous)
            }
        }

        /// Attempts to extends set with items from iterator
        pub fn try_extend(
            &mut self,
            iter: impl IntoIterator<Item = T>,
        ) -> Result<(), NonContiguous> {
            iter.into_iter().try_for_each(|item| self.try_push(item))
        }

        /// Attempts to collect a contiguous set from a fallible iterator
        pub fn try_collect<I, E>(iter: I) -> Result<Result<Self, E>, NonContiguous>
        where
            I: IntoIterator<Item = Result<T, E>>,
        {
            let iter = iter.into_iter();
            let mut c = Self::with_capacity(iter.size_hint().0);

            for item in iter {
                match item {
                    Ok(item) => c.try_push(item)?,
                    Err(err) => return Ok(Err(err)),
                }
            }
            Ok(Ok(c))
        }
    }

    impl<const MAX: usize, T: Adjacent> std::ops::Deref for Contiguous<MAX, T> {
        type Target = [T];

        fn deref(&self) -> &[T] {
            self.items.as_slice()
        }
    }

    impl<const MAX: usize, T: Adjacent> From<Contiguous<MAX, T>> for Vec<T> {
        fn from(contiguous: Contiguous<MAX, T>) -> Self {
            contiguous.items
        }
    }

    impl<const MAX: usize, T: Adjacent> From<Contiguous<MAX, T>> for std::collections::VecDeque<T> {
        fn from(contiguous: Contiguous<MAX, T>) -> Self {
            contiguous.items.into()
        }
    }

    impl<const MAX: usize, T: Adjacent> TryFrom<Vec<T>> for Contiguous<MAX, T> {
        type Error = NonContiguous;

        fn try_from(items: Vec<T>) -> Result<Self, Self::Error> {
            (items.len() <= MAX && is_contiguous(&items))
                .then_some(Self { items })
                .ok_or(NonContiguous)
        }
    }

    /// Attempted to insert a non-contiguous item into a set
    #[derive(Copy, Clone, Debug)]
    pub struct NonContiguous;

    impl std::error::Error for NonContiguous {}

    impl std::fmt::Display for NonContiguous {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            "item is non-contiguous".fmt(f)
        }
    }

    fn is_contiguous<'t, T: Adjacent + 't>(items: impl IntoIterator<Item = &'t T>) -> bool {
        and_previous(items).all(|(prev, item)| match prev {
            Some(prev) => item.is_next(prev),
            None => item.valid_first(),
        })
    }

    fn and_previous<T: Copy>(
        iter: impl IntoIterator<Item = T>,
    ) -> impl Iterator<Item = (Option<T>, T)> {
        let mut previous = None;
        iter.into_iter().map(move |i| (previous.replace(i), i))
    }
}

/// A CUESHEET metadata block
///
/// A cue sheet stores a disc's original layout
/// with all its tracks, index points, and disc-specific metadata.
///
/// This block may occur multiple times in a FLAC file, theoretically.
///
/// | Bits  | Field | Meaning |
/// |------:|------:|---------|
/// | 128×8 | `catalog_number` | media catalog number, in ASCII
/// | 64 | `lead_in_samples` | number of lead-in samples
/// | 1  | `is_cdda` | whether cuesheet corresponds to CD-DA
/// | 7+258×8 | padding | all 0 bits
/// | 8  | track count | number of cuesheet tracks
/// | | `tracks` | cuesheet track₀, cuesheet track₁, …
///
/// Although the structure of this block is not particularly
/// complicated, a CUESHEET block must abide by many rules
/// in order to be considered valid.  Many of these
/// rules are encoded into the type system.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Cuesheet {
    /// A CD-DA Cuesheet, for audio CDs
    CDDA {
        /// Media catalog number in ASCII digits
        ///
        /// For CD-DA, if present, this number must
        /// be exactly 13 ASCII digits followed by all
        /// 0 bytes.
        catalog_number: Option<[cuesheet::Digit; 13]>,

        /// Number of lead-in samples
        ///
        /// For CD-DA, this must be at least 2 seconds
        /// (88200 samples), but may be longer.
        ///
        /// Non-CD-DA cuesheets must always use 0 for
        /// lead-in samples, which is why that variant
        /// does not have this field.
        lead_in_samples: u64,

        /// The cue sheet's non-lead-out tracks
        ///
        /// For CD-DA, 0 ≤ track count ≤ 99
        tracks: contiguous::Contiguous<99, cuesheet::TrackCDDA>,

        /// The required lead-out-track
        ///
        /// This has a track number of 170 and
        /// indicates the end of the disc.
        lead_out: cuesheet::LeadOutCDDA,
    },
    /// A Non-CD-DA Cuesheet, for non-audio CDs
    NonCDDA {
        /// Media catalog number in ASCII digits
        ///
        /// 0 ≤ catalog number digits < 120
        catalog_number: Vec<cuesheet::Digit>,

        /// The cue sheet's non-lead-out tracks
        ///
        /// For Non-CD-DA, 0 ≤ track count ≤ 254
        tracks: contiguous::Contiguous<254, cuesheet::TrackNonCDDA>,

        /// The required lead-out-track
        ///
        /// This has a track number of 255 and
        /// indicates the end of the disc.
        lead_out: cuesheet::LeadOutNonCDDA,
    },
}

impl Cuesheet {
    /// Default number of lead-in samples
    const LEAD_IN: u64 = 44100 * 2;

    /// Maximum catalog number length, in digits
    const CATALOG_LEN: usize = 128;

    /// Media catalog number
    pub fn catalog_number(&self) -> impl std::fmt::Display {
        use cuesheet::Digit;

        enum Digits<'d> {
            Digits(&'d [Digit]),
            Empty,
        }

        impl std::fmt::Display for Digits<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    Self::Digits(digits) => digits.iter().try_for_each(|d| d.fmt(f)),
                    Self::Empty => Ok(()),
                }
            }
        }

        match self {
            Self::CDDA {
                catalog_number: Some(catalog_number),
                ..
            } => Digits::Digits(catalog_number),
            Self::CDDA {
                catalog_number: None,
                ..
            } => Digits::Empty,
            Self::NonCDDA { catalog_number, .. } => Digits::Digits(catalog_number),
        }
    }

    /// The number of lead-in samples for CD-DA discs
    pub fn lead_in_samples(&self) -> Option<u64> {
        match self {
            Self::CDDA {
                lead_in_samples, ..
            } => Some(*lead_in_samples),
            Self::NonCDDA { .. } => None,
        }
    }

    /// If this is a CD-DA cuesheet
    pub fn is_cdda(&self) -> bool {
        matches!(self, Self::CDDA { .. })
    }

    /// Returns total number of tracks in cuesheet
    pub fn track_count(&self) -> usize {
        match self {
            Self::CDDA { tracks, .. } => tracks.len() + 1,
            Self::NonCDDA { tracks, .. } => tracks.len() + 1,
        }
    }

    /// Iterates over all tracks in cuesheet
    ///
    /// Tracks are converted into a unified format suitable for display
    pub fn tracks(&self) -> Box<dyn Iterator<Item = cuesheet::TrackGeneric> + '_> {
        use cuesheet::{Index, Track};

        match self {
            Self::CDDA {
                tracks, lead_out, ..
            } => Box::new(
                tracks
                    .iter()
                    .map(|track| Track {
                        offset: track.offset.into(),
                        number: Some(track.number.get()),
                        isrc: track.isrc.clone(),
                        non_audio: track.non_audio,
                        pre_emphasis: track.pre_emphasis,
                        index_points: track
                            .index_points
                            .iter()
                            .map(|point| Index {
                                number: point.number,
                                offset: point.offset.into(),
                            })
                            .collect(),
                    })
                    .chain(std::iter::once(Track {
                        offset: lead_out.offset.into(),
                        number: None,
                        isrc: lead_out.isrc.clone(),
                        non_audio: lead_out.non_audio,
                        pre_emphasis: lead_out.pre_emphasis,
                        index_points: vec![],
                    })),
            ),
            Self::NonCDDA {
                tracks, lead_out, ..
            } => Box::new(
                tracks
                    .iter()
                    .map(|track| Track {
                        offset: track.offset,
                        number: Some(track.number.get()),
                        isrc: track.isrc.clone(),
                        non_audio: track.non_audio,
                        pre_emphasis: track.pre_emphasis,
                        index_points: track
                            .index_points
                            .iter()
                            .map(|point| Index {
                                number: point.number,
                                offset: point.offset,
                            })
                            .collect(),
                    })
                    .chain(std::iter::once(Track {
                        offset: lead_out.offset,
                        number: None,
                        isrc: lead_out.isrc.clone(),
                        non_audio: lead_out.non_audio,
                        pre_emphasis: lead_out.pre_emphasis,
                        index_points: vec![],
                    })),
            ),
        }
    }

    /// Given a filename to use, returns cuesheet data as text
    pub fn display(&self, filename: &str) -> impl std::fmt::Display {
        struct DisplayCuesheet<'c, 'f> {
            cuesheet: &'c Cuesheet,
            filename: &'f str,
        }

        impl std::fmt::Display for DisplayCuesheet<'_, '_> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                /// A CUESHEET timestamp in MM:SS:FF format
                #[derive(Copy, Clone)]
                pub struct Timestamp {
                    minutes: u64,
                    seconds: u8,
                    frames: u8,
                }

                impl Timestamp {
                    const FRAMES_PER_SECOND: u64 = 75;
                    const SECONDS_PER_MINUTE: u64 = 60;
                    const SAMPLES_PER_FRAME: u64 = 44100 / 75;
                }

                impl From<u64> for Timestamp {
                    fn from(offset: u64) -> Self {
                        let total_frames = offset / Self::SAMPLES_PER_FRAME;

                        Self {
                            minutes: (total_frames / Self::FRAMES_PER_SECOND)
                                / Self::SECONDS_PER_MINUTE,
                            seconds: ((total_frames / Self::FRAMES_PER_SECOND)
                                % Self::SECONDS_PER_MINUTE)
                                .try_into()
                                .unwrap(),
                            frames: (total_frames % Self::FRAMES_PER_SECOND).try_into().unwrap(),
                        }
                    }
                }

                impl std::fmt::Display for Timestamp {
                    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(
                            f,
                            "{:02}:{:02}:{:02}",
                            self.minutes, self.seconds, self.frames
                        )
                    }
                }

                writeln!(f, "FILE \"{}\" FLAC", self.filename)?;

                match self.cuesheet {
                    Cuesheet::CDDA { tracks, .. } => {
                        for track in tracks.iter() {
                            writeln!(
                                f,
                                "  TRACK {} {}",
                                track.number,
                                if track.non_audio {
                                    "NON_AUDIO"
                                } else {
                                    "AUDIO"
                                }
                            )?;
                            for index in track.index_points.iter() {
                                writeln!(
                                    f,
                                    "    INDEX {:02} {}",
                                    index.number,
                                    Timestamp::from(u64::from(index.offset + track.offset)),
                                )?;
                            }
                        }
                    }
                    Cuesheet::NonCDDA { tracks, .. } => {
                        for track in tracks.iter() {
                            writeln!(
                                f,
                                "  TRACK {} {}",
                                track.number,
                                if track.non_audio {
                                    "NON_AUDIO"
                                } else {
                                    "AUDIO"
                                }
                            )?;
                            for index in track.index_points.iter() {
                                writeln!(
                                    f,
                                    "    INDEX {:02} {}",
                                    index.number,
                                    Timestamp::from(index.offset + track.offset),
                                )?;
                            }
                        }
                    }
                }

                Ok(())
            }
        }

        DisplayCuesheet {
            cuesheet: self,
            filename,
        }
    }

    /// Attempts to parse new `Cuesheet` from cue sheet file
    ///
    /// `total_samples` should be the total number
    /// of channel-independent samples, used to
    /// calculate the lead-out track
    ///
    /// `cuesheet` is the entire cuesheet as a string slice
    ///
    /// This is a simplistic cuesheet parser sufficient
    /// for generating FLAC-compatible CUESHEET metadata blocks.
    ///
    /// # Example File
    ///
    /// ```text
    /// FILE "cdimage.wav" WAVE
    ///   TRACK 01 AUDIO
    ///     INDEX 01 00:00:00
    ///   TRACK 02 AUDIO
    ///     INDEX 00 02:57:52
    ///     INDEX 01 03:00:02
    ///   TRACK 03 AUDIO
    ///     INDEX 00 04:46:17
    ///     INDEX 01 04:48:64
    ///   TRACK 04 AUDIO
    ///     INDEX 00 07:09:01
    ///     INDEX 01 07:11:49
    ///   TRACK 05 AUDIO
    ///     INDEX 00 09:11:47
    ///     INDEX 01 09:13:54
    ///   TRACK 06 AUDIO
    ///     INDEX 00 11:10:13
    ///     INDEX 01 11:12:51
    ///   TRACK 07 AUDIO
    ///     INDEX 00 13:03:74
    ///     INDEX 01 13:07:19
    /// ```
    ///
    /// `INDEX` points are in the format:
    ///
    /// ```text
    /// minutes      frames
    ///      ↓↓      ↓↓
    ///      MM::SS::FF
    ///          ↑↑
    ///     seconds
    /// ```
    ///
    /// There are 75 frames per second, and 60 seconds per minute.
    /// Since CD audio has 44100 channel-independent samples per second,
    /// the number of channel-independent samples per frame is 588
    /// (44100 ÷ 75 = 588).
    ///
    /// Thus, the sample offset of each `INDEX` point can be calculated like:
    ///
    /// samples = ((MM × 60 × 75) + (SS × 75) + FF) × 588
    ///
    /// Note that the `INDEX` points are stored in increasing order,
    /// as a standard single file cue sheet.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{Cuesheet, cuesheet::{Track, Index, ISRC}};
    ///
    /// let file = "FILE \"cdimage.wav\" WAVE
    ///   TRACK 01 AUDIO
    ///     INDEX 01 00:00:00
    ///   TRACK 02 AUDIO
    ///     INDEX 00 02:57:52
    ///     INDEX 01 03:00:02
    ///   TRACK 03 AUDIO
    ///     INDEX 00 04:46:17
    ///     INDEX 01 04:48:64
    ///   TRACK 04 AUDIO
    ///     INDEX 00 07:09:01
    ///     INDEX 01 07:11:49
    ///   TRACK 05 AUDIO
    ///     INDEX 00 09:11:47
    ///     INDEX 01 09:13:54
    ///   TRACK 06 AUDIO
    ///     INDEX 00 11:10:13
    ///     INDEX 01 11:12:51
    ///   TRACK 07 AUDIO
    ///     INDEX 00 13:03:74
    ///     INDEX 01 13:07:19
    /// ";
    ///
    /// let cuesheet = Cuesheet::parse(39731748, file).unwrap();
    /// assert!(cuesheet.is_cdda());
    ///
    /// let mut tracks = cuesheet.tracks();
    ///
    /// assert_eq!(
    ///     tracks.next(),
    ///     Some(Track {
    ///         offset: 0,
    ///         number: Some(01),
    ///         isrc: ISRC::None,
    ///         non_audio: false,
    ///         pre_emphasis: false,
    ///         index_points: vec![
    ///             Index { number: 01, offset: 0 },
    ///         ],
    ///     }),
    /// );
    ///
    /// assert_eq!(
    ///     tracks.next(),
    ///     Some(Track {
    ///         // track's offset is that of its first index point
    ///         offset: ((2 * 60 * 75) + (57 * 75) + 52) * 588,
    ///         number: Some(02),
    ///         isrc: ISRC::None,
    ///         non_audio: false,
    ///         pre_emphasis: false,
    ///         index_points: vec![
    ///             // index point offsets are stored relative
    ///             // to the track's offset
    ///             Index { number: 00, offset: 0 },
    ///             Index { number: 01, offset: 175 * 588 }
    ///         ],
    ///     }),
    /// );
    ///
    /// assert_eq!(
    ///     tracks.next(),
    ///     Some(Track {
    ///         offset: ((4 * 60 * 75) + (46 * 75) + 17) * 588,
    ///         number: Some(03),
    ///         isrc: ISRC::None,
    ///         non_audio: false,
    ///         pre_emphasis: false,
    ///         index_points: vec![
    ///             Index { number: 00, offset: 0 },
    ///             Index { number: 01, offset: 197 * 588 }
    ///         ],
    ///     }),
    /// );
    ///
    /// // skip over some tracks for brevity
    /// assert_eq!(tracks.next().and_then(|track| track.number), Some(04));
    /// assert_eq!(tracks.next().and_then(|track| track.number), Some(05));
    /// assert_eq!(tracks.next().and_then(|track| track.number), Some(06));
    /// assert_eq!(tracks.next().and_then(|track| track.number), Some(07));
    ///
    /// // the final lead-out track has an offset of the stream's total samples
    /// // and no index points
    /// assert_eq!(
    ///     tracks.next(),
    ///     Some(Track {
    ///         offset: 39731748,
    ///         number: None,
    ///         isrc: ISRC::None,
    ///         non_audio: false,
    ///         pre_emphasis: false,
    ///         index_points: vec![],
    ///     }),
    /// );
    ///
    /// assert!(tracks.next().is_none());
    /// ```
    pub fn parse(total_samples: u64, cuesheet: &str) -> Result<Self, CuesheetError> {
        use cuesheet::Digit;

        fn cdda_catalog(s: &str) -> Result<Option<[Digit; 13]>, CuesheetError> {
            s.chars()
                .map(Digit::try_from)
                .collect::<Result<Vec<_>, _>>()
                .and_then(|v| {
                    <[Digit; 13] as TryFrom<Vec<Digit>>>::try_from(v)
                        .map_err(|_| CuesheetError::InvalidCatalogNumber)
                })
                .map(Some)
        }

        fn non_cdda_catalog(s: &str) -> Result<Vec<Digit>, CuesheetError> {
            s.chars()
                .map(Digit::try_from)
                .collect::<Result<Vec<_>, _>>()
                .and_then(|v| {
                    (v.len() <= Cuesheet::CATALOG_LEN)
                        .then_some(v)
                        .ok_or(CuesheetError::InvalidCatalogNumber)
                })
        }

        match total_samples.try_into() {
            // if total samples is divisible by 588,
            // try to parse a CDDA cuesheet
            Ok(lead_out_offset) => ParsedCuesheet::parse(cuesheet, cdda_catalog).and_then(
                |ParsedCuesheet {
                     catalog_number,
                     tracks,
                 }| {
                    Ok(Self::CDDA {
                        catalog_number,
                        lead_in_samples: Self::LEAD_IN,
                        lead_out: cuesheet::LeadOutCDDA::new(tracks.last(), lead_out_offset)?,
                        tracks,
                    })
                },
            ),
            // if total samples isn't divisible by 588,
            // only try a non-CDDA cuesheet
            Err(_) => ParsedCuesheet::parse(cuesheet, non_cdda_catalog).and_then(
                |ParsedCuesheet {
                     catalog_number,
                     tracks,
                 }| {
                    Ok(Self::NonCDDA {
                        catalog_number,
                        lead_out: cuesheet::LeadOutNonCDDA::new(tracks.last(), total_samples)?,
                        tracks,
                    })
                },
            ),
        }
    }

    fn track_offsets(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            Self::CDDA {
                tracks, lead_out, ..
            } => Box::new(
                tracks
                    .iter()
                    .map(|t| u64::from(t.offset + *t.index_points.start()))
                    .chain(std::iter::once(u64::from(lead_out.offset))),
            ),
            Self::NonCDDA {
                tracks, lead_out, ..
            } => Box::new(
                tracks
                    .iter()
                    .map(|t| t.offset + t.index_points.start())
                    .chain(std::iter::once(lead_out.offset)),
            ),
        }
    }

    /// Iterates over track ranges in channel-indepedent samples
    ///
    /// Note that the range of each track is from the track
    /// start to the start of the next track, which is indicated
    /// by `INDEX 01`.
    /// It is *not* from the start of the pre-gaps (`INDEX 00`),
    /// which may not be present.
    ///
    /// ```
    /// use flac_codec::metadata::Cuesheet;
    ///
    /// let file = "FILE \"cdimage.wav\" WAVE
    ///   TRACK 01 AUDIO
    ///     INDEX 01 00:00:00
    ///   TRACK 02 AUDIO
    ///     INDEX 00 02:57:52
    ///     INDEX 01 03:00:02
    ///   TRACK 03 AUDIO
    ///     INDEX 00 04:46:17
    ///     INDEX 01 04:48:64
    ///   TRACK 04 AUDIO
    ///     INDEX 00 07:09:01
    ///     INDEX 01 07:11:49
    ///   TRACK 05 AUDIO
    ///     INDEX 00 09:11:47
    ///     INDEX 01 09:13:54
    ///   TRACK 06 AUDIO
    ///     INDEX 00 11:10:13
    ///     INDEX 01 11:12:51
    ///   TRACK 07 AUDIO
    ///     INDEX 00 13:03:74
    ///     INDEX 01 13:07:19
    /// ";
    ///
    /// let cuesheet = Cuesheet::parse(39731748, file).unwrap();
    /// let mut track_ranges = cuesheet.track_sample_ranges();
    ///
    /// // 00:00:00 to 03:00:02
    /// assert_eq!(
    ///     track_ranges.next(),
    ///     Some(0..((3 * 60 * 75) + (0 * 75) + 2) * 588)
    /// );
    ///
    /// // 03:00:02 to 04:48:64
    /// assert_eq!(
    ///     track_ranges.next(),
    ///     Some(((3 * 60 * 75) + (0 * 75) + 2) * 588..((4 * 60 * 75) + (48 * 75) + 64) * 588),
    /// );
    ///
    /// // skip a few tracks for brevity
    /// assert!(track_ranges.next().is_some()); // to 07:11.49
    /// assert!(track_ranges.next().is_some()); // to 09:13:54
    /// assert!(track_ranges.next().is_some()); // to 11:12:51
    /// assert!(track_ranges.next().is_some()); // to 13:07:19
    ///
    /// // 13:07:19 to the lead-out
    /// assert_eq!(
    ///     track_ranges.next(),
    ///     Some(((13 * 60 * 75) + (7 * 75) + 19) * 588..39731748),
    /// );
    ///
    /// assert!(track_ranges.next().is_none());
    /// ```
    pub fn track_sample_ranges(&self) -> impl Iterator<Item = std::ops::Range<u64>> {
        self.track_offsets()
            .zip(self.track_offsets().skip(1))
            .map(|(s, e)| s..e)
    }

    /// Iterates over track ranges in bytes
    ///
    /// Much like [`Cuesheet::track_sample_ranges`], but takes
    /// a channel count and bits-per-sample to convert the ranges to bytes.
    ///
    /// For CD-DA, those values are 2 and 16, respectively.
    ///
    /// # Panics
    ///
    /// Panics if either `channel_count` or `bits_per_sample` are 0
    pub fn track_byte_ranges(
        &self,
        channel_count: u8,
        bits_per_sample: u32,
    ) -> impl Iterator<Item = std::ops::Range<u64>> {
        assert!(channel_count > 0, "channel_count must be > 0");
        assert!(bits_per_sample > 0, "bits_per_sample > 0");

        let multiplier = u64::from(channel_count) * u64::from(bits_per_sample.div_ceil(8));

        self.track_sample_ranges()
            .map(move |std::ops::Range { start, end }| start * multiplier..end * multiplier)
    }
}

block!(Cuesheet, Cuesheet, true);
optional_block!(Cuesheet, Cuesheet);

impl FromBitStream for Cuesheet {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let catalog_number: [u8; Self::CATALOG_LEN] = r.read_to()?;
        let lead_in_samples: u64 = r.read_to()?;
        let is_cdda = r.read_bit()?;
        r.skip(7 + 258 * 8)?;
        let track_count: u8 = r.read_to()?;

        Ok(if is_cdda {
            Self::CDDA {
                catalog_number: {
                    match trim_nulls(&catalog_number) {
                        [] => None,
                        number => Some(
                            number
                                .iter()
                                .copied()
                                .map(cuesheet::Digit::try_from)
                                .collect::<Result<Vec<_>, u8>>()
                                // any of the digits aren't valid ASCII
                                .map_err(|_| Error::from(CuesheetError::InvalidCatalogNumber))?
                                .try_into()
                                // the number isn't the correct size
                                .map_err(|_| Error::from(CuesheetError::InvalidCatalogNumber))?,
                        ),
                    }
                },
                lead_in_samples,
                tracks: contiguous::Contiguous::try_collect(
                    (0..track_count
                        .checked_sub(1)
                        .filter(|c| *c <= 99)
                        .ok_or(Error::from(CuesheetError::NoTracks))?)
                        .map(|_| r.parse()),
                )
                .map_err(|_| Error::from(CuesheetError::TracksOutOfSequence))??,
                lead_out: r.parse()?,
            }
        } else {
            Self::NonCDDA {
                catalog_number: trim_nulls(&catalog_number)
                    .iter()
                    .copied()
                    .map(cuesheet::Digit::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| Error::from(CuesheetError::InvalidCatalogNumber))?,
                tracks: contiguous::Contiguous::try_collect(
                    (0..track_count
                        .checked_sub(1)
                        .ok_or(Error::from(CuesheetError::NoTracks))?)
                        .map(|_| r.parse()),
                )
                .map_err(|_| Error::from(CuesheetError::TracksOutOfSequence))??,
                lead_out: r.parse()?,
            }
        })
    }
}

impl ToBitStream for Cuesheet {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        match self {
            Self::CDDA {
                catalog_number,
                lead_in_samples,
                tracks,
                lead_out,
            } => {
                w.write_from(match catalog_number {
                    Some(number) => {
                        let mut catalog_number = [0; Self::CATALOG_LEN];
                        catalog_number
                            .iter_mut()
                            .zip(number)
                            .for_each(|(o, i)| *o = (*i).into());
                        catalog_number
                    }
                    None => [0; Self::CATALOG_LEN],
                })?;
                w.write_from(*lead_in_samples)?;
                w.write_bit(true)?; // is CD-DA
                w.pad(7 + 258 * 8)?;
                w.write::<8, _>(u8::try_from(tracks.len() + 1).unwrap())?;
                for track in tracks.iter() {
                    w.build(track)?;
                }
                w.build(lead_out)
            }
            Self::NonCDDA {
                catalog_number,
                tracks,
                lead_out,
            } => {
                w.write_from({
                    let mut number = [0; Self::CATALOG_LEN];
                    number
                        .iter_mut()
                        .zip(catalog_number)
                        .for_each(|(o, i)| *o = (*i).into());
                    number
                })?;
                w.write_from::<u64>(0)?; // non-CDDA cuesheets have no lead-in samples
                w.write_bit(false)?; // not CD-DA
                w.pad(7 + 258 * 8)?;
                w.write::<8, _>(u8::try_from(tracks.len() + 1).unwrap())?;
                for track in tracks.iter() {
                    w.build(track)?;
                }
                w.build(lead_out)
            }
        }
    }
}

// trims any trailing null bytes
fn trim_nulls(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., 0] = s {
        s = rest;
    }
    s
}

type ParsedCuesheetTrack<const INDEX_MAX: usize, O> =
    cuesheet::Track<O, NonZero<u8>, cuesheet::IndexVec<INDEX_MAX, O>>;

struct ParsedCuesheet<const TRACK_MAX: usize, const INDEX_MAX: usize, C, O: contiguous::Adjacent> {
    catalog_number: C,
    tracks: contiguous::Contiguous<TRACK_MAX, ParsedCuesheetTrack<INDEX_MAX, O>>,
}

impl<const TRACK_MAX: usize, const INDEX_MAX: usize, C, O>
    ParsedCuesheet<TRACK_MAX, INDEX_MAX, C, O>
where
    C: Default,
    O: contiguous::Adjacent
        + std::str::FromStr
        + Into<u64>
        + std::ops::Sub<Output = O>
        + Default
        + Copy,
{
    fn parse(
        cuesheet: &str,
        parse_catalog: impl Fn(&str) -> Result<C, CuesheetError>,
    ) -> Result<Self, CuesheetError> {
        type WipTrack<const INDEX_MAX: usize, O> = cuesheet::Track<
            Option<O>,
            NonZero<u8>,
            contiguous::Contiguous<INDEX_MAX, cuesheet::Index<O>>,
        >;

        impl<const INDEX_MAX: usize, O: contiguous::Adjacent> WipTrack<INDEX_MAX, O> {
            fn new(number: NonZero<u8>) -> Self {
                Self {
                    offset: None,
                    number,
                    isrc: cuesheet::ISRC::None,
                    non_audio: false,
                    pre_emphasis: false,
                    index_points: contiguous::Contiguous::default(),
                }
            }
        }

        impl<const INDEX_MAX: usize, O: contiguous::Adjacent> TryFrom<WipTrack<INDEX_MAX, O>>
            for ParsedCuesheetTrack<INDEX_MAX, O>
        {
            type Error = CuesheetError;

            fn try_from(track: WipTrack<INDEX_MAX, O>) -> Result<Self, Self::Error> {
                // completed tracks need an offset which
                // is set by adding the first index point
                Ok(Self {
                    offset: track.offset.ok_or(CuesheetError::InvalidTrack)?,
                    number: track.number,
                    isrc: track.isrc,
                    non_audio: track.non_audio,
                    pre_emphasis: track.pre_emphasis,
                    index_points: track.index_points.try_into()?,
                })
            }
        }

        // a bit of a hack, but should be good enough for now
        fn unquote(s: &str) -> &str {
            if s.len() > 1 && s.starts_with('"') && s.ends_with('"') {
                &s[1..s.len() - 1]
            } else {
                s
            }
        }

        let mut wip_track: Option<WipTrack<INDEX_MAX, O>> = None;

        let mut parsed = ParsedCuesheet {
            catalog_number: None,
            tracks: contiguous::Contiguous::default(),
        };

        for line in cuesheet.lines() {
            let line = line.trim();
            match line.split_once(' ').unwrap_or((line, "")) {
                ("CATALOG", "") => return Err(CuesheetError::CatalogMissingNumber),
                ("CATALOG", number) => match parsed.catalog_number {
                    Some(_) => return Err(CuesheetError::MultipleCatalogNumber),
                    ref mut num @ None => {
                        *num = Some(parse_catalog(unquote(number))?);
                    }
                },
                ("TRACK", rest) => {
                    if let Some(finished) = wip_track.replace(WipTrack::new(
                        rest.split_once(' ')
                            .ok_or(CuesheetError::InvalidTrack)?
                            .0
                            .parse()
                            .map_err(|_| CuesheetError::InvalidTrack)?,
                    )) {
                        parsed
                            .tracks
                            .try_push(finished.try_into()?)
                            .map_err(|_| CuesheetError::TracksOutOfSequence)?
                    }
                }
                ("INDEX", rest) => {
                    let (number, offset) = rest
                        .split_once(' ')
                        .ok_or(CuesheetError::InvalidIndexPoint)?;

                    let number: u8 = number
                        .parse()
                        .map_err(|_| CuesheetError::InvalidIndexPoint)?;

                    let offset: O = offset
                        .parse()
                        .map_err(|_| CuesheetError::InvalidIndexPoint)?;

                    let wip_track = wip_track.as_mut().ok_or(CuesheetError::PrematureIndex)?;

                    let index = match &mut wip_track.offset {
                        // work-in progress track has no offset,
                        // so we set it from this index point's
                        // and set the index point's offset to 0
                        track_offset @ None => {
                            // the first index of the first track must have an offset of 0
                            if parsed.tracks.is_empty() && offset.into() != 0 {
                                return Err(CuesheetError::NonZeroFirstIndex);
                            }

                            *track_offset = Some(offset);

                            cuesheet::Index {
                                number,
                                offset: O::default(),
                            }
                        }
                        Some(track_offset) => {
                            // work-in-progress track has offset,
                            // so deduct that offset from this index point's

                            cuesheet::Index {
                                number,
                                offset: offset - *track_offset,
                            }
                        }
                    };

                    wip_track
                        .index_points
                        .try_push(index)
                        .map_err(|_| CuesheetError::IndexPointsOutOfSequence)?;
                }
                ("ISRC", isrc) => {
                    use cuesheet::ISRC;

                    let wip_track = wip_track.as_mut().ok_or(CuesheetError::PrematureISRC)?;

                    if !wip_track.index_points.is_empty() {
                        return Err(CuesheetError::LateISRC)?;
                    }

                    match &mut wip_track.isrc {
                        track_isrc @ ISRC::None => {
                            *track_isrc = ISRC::String(
                                unquote(isrc)
                                    .parse()
                                    .map_err(|_| CuesheetError::InvalidISRC)?,
                            );
                        }
                        ISRC::String(_) => return Err(CuesheetError::MultipleISRC),
                    }
                }
                ("FLAGS", "PRE") => {
                    let wip_track = wip_track.as_mut().ok_or(CuesheetError::PrematureFlags)?;

                    if !wip_track.index_points.is_empty() {
                        return Err(CuesheetError::LateFlags)?;
                    } else {
                        wip_track.pre_emphasis = true;
                    }
                }
                _ => { /*do nothing for now*/ }
            }
        }

        parsed
            .tracks
            .try_push(
                wip_track
                    .take()
                    .ok_or(CuesheetError::NoTracks)?
                    .try_into()?,
            )
            .map_err(|_| CuesheetError::TracksOutOfSequence)?;

        Ok(ParsedCuesheet {
            catalog_number: parsed.catalog_number.unwrap_or_default(),
            tracks: parsed.tracks,
        })
    }
}

/// An error when trying to parse cue sheet data
#[derive(Debug)]
#[non_exhaustive]
pub enum CuesheetError {
    /// CATALOG tag missing catalog number
    CatalogMissingNumber,
    /// multiple CATALOG numbers found
    MultipleCatalogNumber,
    /// invalid CATALOG number
    InvalidCatalogNumber,
    /// Multiple ISRC numbers
    MultipleISRC,
    /// Invalid ISRC number
    InvalidISRC,
    /// ISRC number found before TRACK
    PrematureISRC,
    /// ISRC number after INDEX points
    LateISRC,
    /// FLAGS seen before TRACK
    PrematureFlags,
    /// FLAGS seen after INDEX points
    LateFlags,
    /// ISRC tag missing number
    ISRCMissingNumber,
    /// Unable to parse TRACK field correctly
    InvalidTrack,
    /// No tracks in cue sheet
    NoTracks,
    /// Non-Zero starting INDEX in first TRACK
    NonZeroStartingIndex,
    /// INDEX point occurs before TRACK
    PrematureIndex,
    /// Invalid INDEX point in cuesheet
    InvalidIndexPoint,
    /// No INDEX Points in track
    NoIndexPoints,
    /// Excessive tracks in cue sheet
    ExcessiveTracks,
    /// INDEX points in track are out of sequence
    IndexPointsOutOfSequence,
    /// TRACK points in CUESHEET are out of sequence
    TracksOutOfSequence,
    /// Lead-out track is not beyond all track indices
    ShortLeadOut,
    /// INDEX points in lead-out TRACK
    IndexPointsInLeadout,
    /// Invalid offset for CD-DA CUESHEET
    InvalidCDDAOffset,
    /// first INDEX of first TRACK doesn't have offset of 0
    NonZeroFirstIndex,
}

impl std::error::Error for CuesheetError {}

impl std::fmt::Display for CuesheetError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::CatalogMissingNumber => "CATALOG tag missing number".fmt(f),
            Self::MultipleCatalogNumber => "multiple CATALOG numbers found".fmt(f),
            Self::InvalidCatalogNumber => "invalid CATALOG number".fmt(f),
            Self::MultipleISRC => "multiple ISRC numbers found for track".fmt(f),
            Self::InvalidISRC => "invalid ISRC number found".fmt(f),
            Self::PrematureISRC => "ISRC number found before TRACK".fmt(f),
            Self::LateISRC => "ISRC number found after INDEX points".fmt(f),
            Self::PrematureFlags => "FLAGS found before TRACK".fmt(f),
            Self::LateFlags => "FLAGS found after INDEX points".fmt(f),
            Self::ISRCMissingNumber => "ISRC tag missing number".fmt(f),
            Self::InvalidTrack => "invalid TRACK entry".fmt(f),
            Self::NoTracks => "no TRACK entries in cue sheet".fmt(f),
            Self::NonZeroStartingIndex => {
                "first INDEX of first track must have 00:00:00 offset".fmt(f)
            }
            Self::PrematureIndex => "INDEX found before TRACK".fmt(f),
            Self::InvalidIndexPoint => "invalid INDEX entry".fmt(f),
            Self::NoIndexPoints => "no INDEX points in track".fmt(f),
            Self::ExcessiveTracks => "excessive tracks in CUESHEET".fmt(f),
            Self::IndexPointsOutOfSequence => "INDEX points out of sequence".fmt(f),
            Self::TracksOutOfSequence => "TRACKS out of sequence".fmt(f),
            Self::ShortLeadOut => "lead-out track not beyond final INDEX point".fmt(f),
            Self::IndexPointsInLeadout => "INDEX points in lead-out TRACK".fmt(f),
            Self::InvalidCDDAOffset => "invalid offset for CD-DA CUESHEET".fmt(f),
            Self::NonZeroFirstIndex => "first index of first track has non-zero offset".fmt(f),
        }
    }
}

/// A PICTURE metadata block
///
/// Picture blocks are for embedding artwork
/// such as album covers, liner notes, etc.
///
/// This block may occur multiple times in a FLAC file.
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 32   | `picture_type` | picture type
/// | 32   | media type len | media type length, in bytes
/// | `media type len`×8 | `media_type` | picture's MIME type
/// | 32   | description len | description length, in bytes
/// | `description len`×8 | `description` | description of picture, in UTF-8
/// | 32   | `width` | width of picture, in pixels
/// | 32   | `height`| height of picture, in pixels
/// | 32   | `color_depth` | color depth of picture in bits-per-pixel
/// | 32   | `colors_used` | for indexed-color pictures, number of colors used
/// | 32   | data len | length of picture data, in bytes
/// | `data len`×8 | `data` | raw picture data
///
/// # Example
/// ```
/// use bitstream_io::{BitReader, BitRead, BigEndian};
/// use flac_codec::metadata::{Picture, PictureType};
///
/// let data: &[u8] = &[
///     0x00, 0x00, 0x00, 0x03,  // picture type
///     0x00, 0x00, 0x00, 0x09,  // media type len (9 bytes)
///     0x69, 0x6d, 0x61, 0x67, 0x65, 0x2f, 0x70, 0x6e, 0x67,
///     0x00, 0x00, 0x00, 0x0a,  // description len (10 bytes)
///     0x54, 0x65, 0x73, 0x74, 0x20, 0x49, 0x6d, 0x61, 0x67, 0x65,
///     0x00, 0x00, 0x00, 0x10,  // width
///     0x00, 0x00, 0x00, 0x09,  // height
///     0x00, 0x00, 0x00, 0x18,  // color depth
///     0x00, 0x00, 0x00, 0x00,  // color count
///     0x00, 0x00, 0x00, 0x5c,  // data len (92 bytes)
///     0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
///     0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
///     0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x09,
///     0x08, 0x02, 0x00, 0x00, 0x00, 0xb4, 0x48, 0x3b,
///     0x65, 0x00, 0x00, 0x00, 0x09, 0x70, 0x48, 0x59,
///     0x73, 0x00, 0x00, 0x2e, 0x23, 0x00, 0x00, 0x2e,
///     0x23, 0x01, 0x78, 0xa5, 0x3f, 0x76, 0x00, 0x00,
///     0x00, 0x0e, 0x49, 0x44, 0x41, 0x54, 0x18, 0xd3,
///     0x63, 0x60, 0x18, 0x05, 0x43, 0x12, 0x00, 0x00,
///     0x01, 0xb9, 0x00, 0x01, 0xed, 0x78, 0x29, 0x25,
///     0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
///     0xae, 0x42, 0x60, 0x82,
/// ];
///
/// let mut r = BitReader::endian(data, BigEndian);
/// assert_eq!(
///     r.parse::<Picture>().unwrap(),
///     Picture {
///         picture_type: PictureType::FrontCover,  // type 3
///         media_type: "image/png".to_owned(),
///         description: "Test Image".to_owned(),
///         width: 0x00_00_00_10,                   // 16 pixels
///         height: 0x00_00_00_09,                  // 9 pixels
///         color_depth: 0x00_00_00_18,             // 24 bits-per-pixel
///         colors_used: None,                      // not indexed
///         data: vec![
///             0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
///             0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
///             0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x09,
///             0x08, 0x02, 0x00, 0x00, 0x00, 0xb4, 0x48, 0x3b,
///             0x65, 0x00, 0x00, 0x00, 0x09, 0x70, 0x48, 0x59,
///             0x73, 0x00, 0x00, 0x2e, 0x23, 0x00, 0x00, 0x2e,
///             0x23, 0x01, 0x78, 0xa5, 0x3f, 0x76, 0x00, 0x00,
///             0x00, 0x0e, 0x49, 0x44, 0x41, 0x54, 0x18, 0xd3,
///             0x63, 0x60, 0x18, 0x05, 0x43, 0x12, 0x00, 0x00,
///             0x01, 0xb9, 0x00, 0x01, 0xed, 0x78, 0x29, 0x25,
///             0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
///             0xae, 0x42, 0x60, 0x82,
///         ],
///     },
/// );
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Picture {
    /// The picture type
    pub picture_type: PictureType,
    /// The media type string as specified by RFC2046
    pub media_type: String,
    /// The description of the picture
    pub description: String,
    /// The width of the picture in pixels
    pub width: u32,
    /// The height of the picture in pixels
    pub height: u32,
    /// The color depth of the picture in bits per pixel
    pub color_depth: u32,
    /// For indexed-color pictures, the number of colors used
    pub colors_used: Option<NonZero<u32>>,
    /// The binary picture data
    pub data: Vec<u8>,
}

block!(Picture, Picture, true);
optional_block!(Picture, Picture);

impl Picture {
    /// Attempt to create a new PICTURE block from raw image data
    ///
    /// Currently supported image types for this method are:
    ///
    /// - JPEG
    /// - PNG
    /// - GIF
    ///
    /// Any type of image data may be placed in a PICTURE block,
    /// but the user may have to use external crates
    /// to determine their proper image metrics
    /// to build a block from.
    ///
    /// # Errors
    ///
    /// Returns an error if some problem occurs reading
    /// or identifying the file.
    pub fn new<S, V>(
        picture_type: PictureType,
        description: S,
        data: V,
    ) -> Result<Self, InvalidPicture>
    where
        S: Into<String>,
        V: Into<Vec<u8>> + AsRef<[u8]>,
    {
        let metrics = PictureMetrics::try_new(data.as_ref())?;
        Ok(Self {
            picture_type,
            description: description.into(),
            data: data.into(),
            media_type: metrics.media_type.to_owned(),
            width: metrics.width,
            height: metrics.height,
            color_depth: metrics.color_depth,
            colors_used: metrics.colors_used,
        })
    }

    /// Attempt to create new PICTURE block from file on disk
    pub fn open<S, P>(
        picture_type: PictureType,
        description: S,
        path: P,
    ) -> Result<Self, InvalidPicture>
    where
        S: Into<String>,
        P: AsRef<Path>,
    {
        std::fs::read(path)
            .map_err(InvalidPicture::Io)
            .and_then(|data| Self::new(picture_type, description, data))
    }
}

impl FromBitStream for Picture {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        fn prefixed_field<R: BitRead + ?Sized>(r: &mut R) -> std::io::Result<Vec<u8>> {
            let size = r.read_to::<u32>()?;
            r.read_to_vec(size.try_into().unwrap())
        }

        Ok(Self {
            picture_type: r.parse()?,
            media_type: String::from_utf8(prefixed_field(r)?)?,
            description: String::from_utf8(prefixed_field(r)?)?,
            width: r.read_to()?,
            height: r.read_to()?,
            color_depth: r.read_to()?,
            colors_used: r.read::<32, _>()?,
            data: prefixed_field(r)?,
        })
    }
}

impl ToBitStream for Picture {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Error> {
        fn prefixed_field<W: BitWrite + ?Sized>(
            w: &mut W,
            field: &[u8],
            error: Error,
        ) -> Result<(), Error> {
            w.write_from::<u32>(field.len().try_into().map_err(|_| error)?)
                .map_err(Error::Io)?;
            w.write_bytes(field).map_err(Error::Io)
        }

        w.build(&self.picture_type)?;
        prefixed_field(w, self.media_type.as_bytes(), Error::ExcessiveStringLength)?;
        prefixed_field(w, self.description.as_bytes(), Error::ExcessiveStringLength)?;
        w.write_from(self.width)?;
        w.write_from(self.height)?;
        w.write_from(self.color_depth)?;
        w.write::<32, _>(self.colors_used)?;
        prefixed_field(w, &self.data, Error::ExcessivePictureSize)
    }
}

/// Defined variants of PICTURE type
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PictureType {
    /// Other
    Other = 0,
    /// PNG file icon of 32x32 pixels
    Png32x32 = 1,
    /// General file icon
    GeneralFileIcon = 2,
    /// Front cover
    FrontCover = 3,
    /// Back cover
    BackCover = 4,
    /// Liner notes page
    LinerNotes = 5,
    /// Media label (e.g., CD, Vinyl or Cassette label)
    MediaLabel = 6,
    /// Lead artist, lead performer, or soloist
    LeadArtist = 7,
    /// Artist or performer
    Artist = 8,
    /// Conductor
    Conductor = 9,
    /// Band or orchestra
    Band = 10,
    /// Composer
    Composer = 11,
    /// Lyricist or text writer
    Lyricist = 12,
    /// Recording location
    RecordingLocation = 13,
    /// During recording
    DuringRecording = 14,
    /// During performance
    DuringPerformance = 15,
    /// Movie or video screen capture
    ScreenCapture = 16,
    /// A bright colored fish
    Fish = 17,
    /// Illustration
    Illustration = 18,
    /// Band or artist logotype
    BandLogo = 19,
    /// Publisher or studio logotype
    PublisherLogo = 20,
}

impl std::fmt::Display for PictureType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Other => "Other".fmt(f),
            Self::Png32x32 => "32×32 PNG Icon".fmt(f),
            Self::GeneralFileIcon => "General File Icon".fmt(f),
            Self::FrontCover => "Cover (front)".fmt(f),
            Self::BackCover => "Cover (back)".fmt(f),
            Self::LinerNotes => "Liner Notes".fmt(f),
            Self::MediaLabel => "Media Label".fmt(f),
            Self::LeadArtist => "Lead Artist".fmt(f),
            Self::Artist => "Artist".fmt(f),
            Self::Conductor => "Conductor".fmt(f),
            Self::Band => "Band or Orchestra".fmt(f),
            Self::Composer => "Composer".fmt(f),
            Self::Lyricist => "lyricist or Text Writer".fmt(f),
            Self::RecordingLocation => "Recording Location".fmt(f),
            Self::DuringRecording => "During Recording".fmt(f),
            Self::DuringPerformance => "During Performance".fmt(f),
            Self::ScreenCapture => "Movie or Video Screen Capture".fmt(f),
            Self::Fish => "A Bright Colored Fish".fmt(f),
            Self::Illustration => "Illustration".fmt(f),
            Self::BandLogo => "Band or Artist Logotype".fmt(f),
            Self::PublisherLogo => "Publisher or Studio Logotype".fmt(f),
        }
    }
}

impl FromBitStream for PictureType {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        match r.read_to::<u32>()? {
            0 => Ok(Self::Other),
            1 => Ok(Self::Png32x32),
            2 => Ok(Self::GeneralFileIcon),
            3 => Ok(Self::FrontCover),
            4 => Ok(Self::BackCover),
            5 => Ok(Self::LinerNotes),
            6 => Ok(Self::MediaLabel),
            7 => Ok(Self::LeadArtist),
            8 => Ok(Self::Artist),
            9 => Ok(Self::Conductor),
            10 => Ok(Self::Band),
            11 => Ok(Self::Composer),
            12 => Ok(Self::Lyricist),
            13 => Ok(Self::RecordingLocation),
            14 => Ok(Self::DuringRecording),
            15 => Ok(Self::DuringPerformance),
            16 => Ok(Self::ScreenCapture),
            17 => Ok(Self::Fish),
            18 => Ok(Self::Illustration),
            19 => Ok(Self::BandLogo),
            20 => Ok(Self::PublisherLogo),
            _ => Err(Error::InvalidPictureType),
        }
    }
}

impl ToBitStream for PictureType {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from::<u32>(match self {
            Self::Other => 0,
            Self::Png32x32 => 1,
            Self::GeneralFileIcon => 2,
            Self::FrontCover => 3,
            Self::BackCover => 4,
            Self::LinerNotes => 5,
            Self::MediaLabel => 6,
            Self::LeadArtist => 7,
            Self::Artist => 8,
            Self::Conductor => 9,
            Self::Band => 10,
            Self::Composer => 11,
            Self::Lyricist => 12,
            Self::RecordingLocation => 13,
            Self::DuringRecording => 14,
            Self::DuringPerformance => 15,
            Self::ScreenCapture => 16,
            Self::Fish => 17,
            Self::Illustration => 18,
            Self::BandLogo => 19,
            Self::PublisherLogo => 20,
        })
    }
}

/// An error when trying to identify a picture's metrics
#[derive(Debug)]
#[non_exhaustive]
pub enum InvalidPicture {
    /// An I/O Error
    Io(std::io::Error),
    /// Unsupported Image Format
    Unsupported,
    /// Invalid PNG File
    Png(&'static str),
    /// Invalid JPEG File
    Jpeg(&'static str),
    /// Invalid GIF File
    Gif(&'static str),
}

impl From<std::io::Error> for InvalidPicture {
    #[inline]
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl std::error::Error for InvalidPicture {}

impl std::fmt::Display for InvalidPicture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Io(err) => err.fmt(f),
            Self::Unsupported => "unsupported image format".fmt(f),
            Self::Png(s) => write!(f, "PNG parsing error : {s}"),
            Self::Jpeg(s) => write!(f, "JPEG parsing error : {s}"),
            Self::Gif(s) => write!(f, "GIF parsing error : {s}"),
        }
    }
}

struct PictureMetrics {
    media_type: &'static str,
    width: u32,
    height: u32,
    color_depth: u32,
    colors_used: Option<NonZero<u32>>,
}

impl PictureMetrics {
    fn try_new(data: &[u8]) -> Result<Self, InvalidPicture> {
        if data.starts_with(b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A") {
            Self::try_png(data)
        } else if data.starts_with(b"\xFF\xD8\xFF") {
            Self::try_jpeg(data)
        } else if data.starts_with(b"GIF") {
            Self::try_gif(data)
        } else {
            Err(InvalidPicture::Unsupported)
        }
    }

    fn try_png(data: &[u8]) -> Result<Self, InvalidPicture> {
        // this is an *extremely* cut-down PNG parser
        // that handles just enough of the format to get
        // image metadata, but does *not* validate things
        // like the block CRC32s

        fn plte_colors<R: ByteRead>(mut r: R) -> Result<u32, InvalidPicture> {
            loop {
                let block_len = r.read::<u32>()?;
                match r.read::<[u8; 4]>()?.as_slice() {
                    b"PLTE" => {
                        if block_len % 3 == 0 {
                            break Ok(block_len / 3);
                        } else {
                            break Err(InvalidPicture::Png("invalid PLTE length"));
                        }
                    }
                    _ => {
                        r.skip(block_len)?;
                        let _crc = r.read::<u32>()?;
                    }
                }
            }
        }

        let mut r = ByteReader::endian(data, BigEndian);
        if &r.read::<[u8; 8]>()? != b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A" {
            return Err(InvalidPicture::Png("not a PNG image"));
        }

        // IHDR chunk must be first
        if r.read::<u32>()? != 0x0d {
            return Err(InvalidPicture::Png("invalid IHDR length"));
        }
        if &r.read::<[u8; 4]>()? != b"IHDR" {
            return Err(InvalidPicture::Png("IHDR chunk not first"));
        }
        let width = r.read()?;
        let height = r.read()?;
        let bit_depth = r.read::<u8>()?;
        let color_type = r.read::<u8>()?;
        let _compression_method = r.read::<u8>()?;
        let _filter_method = r.read::<u8>()?;
        let _interlace_method = r.read::<u8>()?;
        let _crc = r.read::<u32>()?;

        let (color_depth, colors_used) = match color_type {
            0 => (bit_depth.into(), None),           // grayscale
            2 => ((bit_depth * 3).into(), None),     // RGB
            3 => (0, NonZero::new(plte_colors(r)?)), // palette
            4 => ((bit_depth * 2).into(), None),     // grayscale + alpha
            6 => ((bit_depth * 4).into(), None),     // RGB + alpha
            _ => return Err(InvalidPicture::Png("invalid color type")),
        };

        Ok(Self {
            media_type: "image/png",
            width,
            height,
            color_depth,
            colors_used,
        })
    }

    fn try_jpeg(data: &[u8]) -> Result<Self, InvalidPicture> {
        let mut r = ByteReader::endian(data, BigEndian);

        if r.read::<u8>()? != 0xFF || r.read::<u8>()? != 0xD8 {
            return Err(InvalidPicture::Jpeg("invalid JPEG marker"));
        }

        loop {
            if r.read::<u8>()? != 0xFF {
                break Err(InvalidPicture::Jpeg("invalid JPEG marker"));
            }
            match r.read::<u8>()? {
                0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD
                | 0xCE | 0xCF => {
                    let _len = r.read::<u16>()?;
                    let data_precision = r.read::<u8>()?;
                    let height = r.read::<u16>()?;
                    let width = r.read::<u16>()?;
                    let components = r.read::<u8>()?;
                    break Ok(Self {
                        media_type: "image/jpeg",
                        width: width.into(),
                        height: height.into(),
                        color_depth: (data_precision * components).into(),
                        colors_used: None,
                    });
                }
                _ => {
                    let segment_length = r
                        .read::<u16>()?
                        .checked_sub(2)
                        .ok_or(InvalidPicture::Jpeg("invalid segment length"))?;
                    r.skip(segment_length.into())?;
                }
            }
        }
    }

    fn try_gif(data: &[u8]) -> Result<Self, InvalidPicture> {
        let mut r = BitReader::endian(data, LittleEndian);

        if &r.read_to::<[u8; 3]>()? != b"GIF" {
            return Err(InvalidPicture::Gif("invalid GIF signature"));
        }

        r.skip(3 * 8)?; // ignore version bytes

        Ok(Self {
            media_type: "image/gif",
            width: r.read::<16, _>()?,
            height: r.read::<16, _>()?,
            colors_used: NonZero::new(1 << (r.read::<3, u32>()? + 1)),
            color_depth: 0,
        })
    }
}

/// A collection of metadata blocks
///
/// This collection enforces the restriction that FLAC files
/// must always contain a STREAMINFO metadata block
/// and that block must always be first in the file.
///
/// Because it is required, that block may be retrieved
/// unconditionally from this collection, while all others
/// are optional and may appear in any order.
#[derive(Clone, Debug)]
pub struct BlockList {
    streaminfo: Streaminfo,
    blocks: Vec<private::OptionalBlock>,
}

impl BlockList {
    /// Creates `BlockList` from initial STREAMINFO
    pub fn new(streaminfo: Streaminfo) -> Self {
        Self {
            streaminfo,
            blocks: Vec::default(),
        }
    }

    /// Reads `BlockList` from the given reader
    ///
    /// This assumes the reader is rewound to the
    /// beginning of the file.
    ///
    /// Because this may perform many small reads,
    /// using a buffered reader is preferred when
    /// reading from a raw file.
    ///
    /// # Errors
    ///
    /// Returns any error reading or parsing metadata blocks
    pub fn read<R: std::io::Read>(r: R) -> Result<Self, Error> {
        // TODO - change this to flatten once that stabilizes
        read_blocks(r).collect::<Result<Result<Self, _>, _>>()?
    }

    /// Reads `BlockList` from the given file path
    ///
    /// # Errors
    ///
    /// Returns any error reading or parsing metadata blocks
    pub fn open<P: AsRef<Path>>(p: P) -> Result<Self, Error> {
        File::open(p.as_ref())
            .map(BufReader::new)
            .map_err(Error::Io)
            .and_then(BlockList::read)
    }

    /// Returns reference to our STREAMINFO metadata block
    pub fn streaminfo(&self) -> &Streaminfo {
        &self.streaminfo
    }

    /// Returns exclusive reference to our STREAMINFO metadata block
    ///
    /// Care must be taken when modifying the STREAMINFO, or
    /// one's file could be rendered unplayable.
    pub fn streaminfo_mut(&mut self) -> &mut Streaminfo {
        &mut self.streaminfo
    }

    /// Iterates over all the metadata blocks
    pub fn blocks(&self) -> impl Iterator<Item = BlockRef<'_>> {
        std::iter::once(self.streaminfo.as_block_ref())
            .chain(self.blocks.iter().map(|b| b.as_block_ref()))
    }

    /// Inserts new optional metadata block
    ///
    /// If the block may only occur once in the stream
    /// (such as the SEEKTABLE), any existing block of
    /// the same type removed and extracted first.
    pub fn insert<B: OptionalMetadataBlock>(&mut self, block: B) -> Option<B> {
        if B::MULTIPLE {
            self.blocks.push(block.into());
            None
        } else {
            match self
                .blocks
                .iter_mut()
                .find_map(|b| B::try_from_opt_block_mut(b).ok())
            {
                Some(b) => Some(std::mem::replace(b, block)),
                None => {
                    self.blocks.push(block.into());
                    None
                }
            }
        }
    }

    /// Gets reference to metadata block, if present
    ///
    /// If the block type occurs multiple times,
    /// this returns the first instance.
    pub fn get<B: OptionalMetadataBlock>(&self) -> Option<&B> {
        self.blocks
            .iter()
            .find_map(|b| B::try_from_opt_block(b).ok())
    }

    /// Gets mutable reference to metadata block, if present
    ///
    /// If the block type occurs multiple times,
    /// this returns the first instance.
    pub fn get_mut<B: OptionalMetadataBlock>(&mut self) -> Option<&mut B> {
        self.blocks
            .iter_mut()
            .find_map(|b| B::try_from_opt_block_mut(b).ok())
    }

    /// Gets mutable references to a pair of metadata blocks
    ///
    /// If either block type occurs multiple times,
    /// this returns the first instance.
    pub fn get_pair_mut<B, C>(&mut self) -> (Option<&mut B>, Option<&mut C>)
    where
        B: OptionalMetadataBlock,
        C: OptionalMetadataBlock,
    {
        use std::ops::ControlFlow;

        match self
            .blocks
            .iter_mut()
            .try_fold((None, None), |acc, block| match acc {
                (first @ None, second @ None) => {
                    ControlFlow::Continue(match B::try_from_opt_block_mut(block) {
                        Ok(first) => (Some(first), second),
                        Err(block) => (first, C::try_from_opt_block_mut(block).ok()),
                    })
                }
                (first @ Some(_), None) => {
                    ControlFlow::Continue((first, C::try_from_opt_block_mut(block).ok()))
                }
                (None, second @ Some(_)) => {
                    ControlFlow::Continue((B::try_from_opt_block_mut(block).ok(), second))
                }
                pair @ (Some(_), Some(_)) => ControlFlow::Break(pair),
            }) {
            ControlFlow::Break(p) | ControlFlow::Continue(p) => p,
        }
    }

    /// Gets references to all metadata blocks of the given type
    pub fn get_all<'b, B: OptionalMetadataBlock + 'b>(&'b self) -> impl Iterator<Item = &'b B> {
        self.blocks
            .iter()
            .filter_map(|b| B::try_from_opt_block(b).ok())
    }

    /// Gets exclusive references to all metadata blocks of the given type
    pub fn get_all_mut<'b, B: OptionalMetadataBlock + 'b>(
        &'b mut self,
    ) -> impl Iterator<Item = &'b mut B> {
        self.blocks
            .iter_mut()
            .filter_map(|b| B::try_from_opt_block_mut(b).ok())
    }

    /// Returns `true` if block exists in list
    pub fn has<B: OptionalMetadataBlock>(&self) -> bool {
        self.get::<B>().is_some()
    }

    /// Removes all instances of the given metadata block type
    pub fn remove<B: OptionalMetadataBlock>(&mut self) {
        self.blocks.retain(|b| b.block_type() != B::TYPE)
    }

    /// Removes and returns all instances of the given block type
    pub fn extract<B: OptionalMetadataBlock>(&mut self) -> impl Iterator<Item = B> {
        self.blocks
            .extract_if(.., |block| block.block_type() == B::TYPE)
            .filter_map(|b| B::try_from(b).ok())
    }

    /// Updates first instance of the given block, creating it if necessary
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{BlockList, Streaminfo, VorbisComment};
    /// use flac_codec::metadata::fields::ARTIST;
    ///
    /// // build a BlockList with a dummy Streaminfo
    /// let mut blocklist = BlockList::new(
    ///     Streaminfo {
    ///         minimum_block_size: 0,
    ///         maximum_block_size: 0,
    ///         minimum_frame_size: None,
    ///         maximum_frame_size: None,
    ///         sample_rate: 44100,
    ///         channels: 1u8.try_into().unwrap(),
    ///         bits_per_sample: 16u32.try_into().unwrap(),
    ///         total_samples: None,
    ///         md5: None,
    ///     },
    /// );
    ///
    /// // the block starts out with no comment
    /// assert!(blocklist.get::<VorbisComment>().is_none());
    ///
    /// // update Vorbis Comment with artist field,
    /// // which adds a new block to the list
    /// blocklist.update::<VorbisComment>(
    ///     |vc| vc.insert(ARTIST, "Artist 1")
    /// );
    /// assert!(blocklist.get::<VorbisComment>().is_some());
    ///
    /// // updating Vorbis Comment again reuses that same block
    /// blocklist.update::<VorbisComment>(
    ///     |vc| vc.insert(ARTIST, "Artist 2")
    /// );
    ///
    /// // the block now has two entries
    /// assert_eq!(
    ///     blocklist.get::<VorbisComment>()
    ///         .unwrap()
    ///         .all(ARTIST).collect::<Vec<_>>(),
    ///     vec!["Artist 1", "Artist 2"],
    /// );
    /// ```
    pub fn update<B>(&mut self, f: impl FnOnce(&mut B))
    where
        B: OptionalMetadataBlock + Default,
    {
        match self.get_mut() {
            Some(block) => f(block),
            None => {
                let mut b = B::default();
                f(&mut b);
                self.blocks.push(b.into());
            }
        }
    }

    /// Sorts optional metadata blocks by block type
    ///
    /// The function converts the type to some key which is
    /// used for ordering blocks from smallest to largest.
    ///
    /// The order of blocks of the same type is preserved.
    /// This is an important consideration for APPLICATION
    /// metadata blocks, which may contain foreign metadata
    /// chunks that must be re-applied in the same order.
    ///
    /// # Example
    ///
    /// ```
    /// use flac_codec::metadata::{
    ///     BlockList, Streaminfo, Application, Padding, AsBlockRef,
    ///     OptionalBlockType,
    /// };
    ///
    /// // build a BlockList with a dummy Streaminfo
    /// let streaminfo = Streaminfo {
    ///     minimum_block_size: 0,
    ///     maximum_block_size: 0,
    ///     minimum_frame_size: None,
    ///     maximum_frame_size: None,
    ///     sample_rate: 44100,
    ///     channels: 1u8.try_into().unwrap(),
    ///     bits_per_sample: 16u32.try_into().unwrap(),
    ///     total_samples: None,
    ///     md5: None,
    /// };
    ///
    /// let mut blocklist = BlockList::new(streaminfo.clone());
    ///
    /// // add some blocks
    /// let application_1 = Application {
    ///     id: 0x1234,
    ///     data: vec![0x01, 0x02, 0x03, 0x04],
    /// };
    /// blocklist.insert(application_1.clone());
    ///
    /// let padding = Padding {
    ///     size: 10u32.try_into().unwrap(),
    /// };
    /// blocklist.insert(padding.clone());
    ///
    /// let application_2 = Application {
    ///     id: 0x6789,
    ///     data: vec![0x06, 0x07, 0x08, 0x09],
    /// };
    /// blocklist.insert(application_2.clone());
    ///
    /// // check their inital order
    /// let mut iter = blocklist.blocks();
    /// assert_eq!(iter.next(), Some(streaminfo.as_block_ref()));
    /// assert_eq!(iter.next(), Some(application_1.as_block_ref()));
    /// assert_eq!(iter.next(), Some(padding.as_block_ref()));
    /// assert_eq!(iter.next(), Some(application_2.as_block_ref()));
    /// assert_eq!(iter.next(), None);
    /// drop(iter);
    ///
    /// // sort the blocks to put padding last
    /// blocklist.sort_by(|t| match t {
    ///     OptionalBlockType::Application => 0,
    ///     OptionalBlockType::SeekTable => 1,
    ///     OptionalBlockType::VorbisComment => 2,
    ///     OptionalBlockType::Cuesheet => 3,
    ///     OptionalBlockType::Picture => 4,
    ///     OptionalBlockType::Padding => 5,
    /// });
    ///
    /// // re-check their new order
    /// let mut iter = blocklist.blocks();
    /// assert_eq!(iter.next(), Some(streaminfo.as_block_ref()));
    /// assert_eq!(iter.next(), Some(application_1.as_block_ref()));
    /// assert_eq!(iter.next(), Some(application_2.as_block_ref()));
    /// assert_eq!(iter.next(), Some(padding.as_block_ref()));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn sort_by<O: Ord>(&mut self, f: impl Fn(OptionalBlockType) -> O) {
        self.blocks
            .sort_by_key(|block| f(block.optional_block_type()));
    }
}

impl Metadata for BlockList {
    fn channel_count(&self) -> u8 {
        self.streaminfo.channels.get()
    }

    fn channel_mask(&self) -> ChannelMask {
        use fields::CHANNEL_MASK;

        self.get::<VorbisComment>()
            .and_then(|c| c.get(CHANNEL_MASK).and_then(|m| m.parse().ok()))
            .unwrap_or(ChannelMask::from_channels(self.channel_count()))
    }

    fn sample_rate(&self) -> u32 {
        self.streaminfo.sample_rate
    }

    fn bits_per_sample(&self) -> u32 {
        self.streaminfo.bits_per_sample.into()
    }

    fn total_samples(&self) -> Option<u64> {
        self.streaminfo.total_samples.map(|s| s.get())
    }

    fn md5(&self) -> Option<&[u8; 16]> {
        self.streaminfo.md5.as_ref()
    }
}

impl FromIterator<Block> for Result<BlockList, Error> {
    fn from_iter<T: IntoIterator<Item = Block>>(iter: T) -> Self {
        let mut iter = iter.into_iter();

        let mut list = match iter.next() {
            Some(Block::Streaminfo(streaminfo)) => BlockList::new(streaminfo),
            Some(_) | None => return Err(Error::MissingStreaminfo),
        };

        for block in iter {
            match block {
                Block::Streaminfo(_) => return Err(Error::MultipleStreaminfo),
                Block::Padding(p) => {
                    list.insert(p);
                }
                Block::Application(p) => {
                    list.insert(p);
                }
                Block::SeekTable(p) => {
                    list.insert(p);
                }
                Block::VorbisComment(p) => {
                    list.insert(p);
                }
                Block::Cuesheet(p) => {
                    list.insert(p);
                }
                Block::Picture(p) => {
                    list.insert(p);
                }
            }
        }

        Ok(list)
    }
}

impl IntoIterator for BlockList {
    type Item = Block;
    type IntoIter = Box<dyn Iterator<Item = Block>>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(
            std::iter::once(self.streaminfo.into())
                .chain(self.blocks.into_iter().map(|b| b.into())),
        )
    }
}

impl<B: OptionalMetadataBlock> Extend<B> for BlockList {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = B>,
    {
        for block in iter {
            self.insert(block);
        }
    }
}

/// A type of FLAC metadata block which is not required
///
/// The STREAMINFO block is required.  All others are optional.
pub trait OptionalMetadataBlock: MetadataBlock + private::OptionalMetadataBlock {
    /// Our optional block type
    const OPTIONAL_TYPE: OptionalBlockType;
}

/// A type of optional FLAC metadata block which is portable
///
/// These are blocks which may be safely ported from one
/// encoding of a FLAC file to another.
///
/// All blocks except STREAMINFO and SEEKTABLE are considered portable.
pub trait PortableMetadataBlock: OptionalMetadataBlock {}

impl PortableMetadataBlock for Padding {}
impl PortableMetadataBlock for Application {}
impl PortableMetadataBlock for VorbisComment {}
impl PortableMetadataBlock for Cuesheet {}
impl PortableMetadataBlock for Picture {}

mod private {
    use super::{
        Application, AsBlockRef, Block, BlockRef, BlockType, Cuesheet, OptionalBlockType, Padding,
        Picture, SeekTable, Streaminfo, VorbisComment,
    };

    #[derive(Clone, Debug)]
    pub enum OptionalBlock {
        Padding(Padding),
        Application(Application),
        SeekTable(SeekTable),
        VorbisComment(VorbisComment),
        Cuesheet(Cuesheet),
        Picture(Picture),
    }

    impl OptionalBlock {
        pub fn block_type(&self) -> BlockType {
            match self {
                Self::Padding(_) => BlockType::Padding,
                Self::Application(_) => BlockType::Application,
                Self::SeekTable(_) => BlockType::SeekTable,
                Self::VorbisComment(_) => BlockType::VorbisComment,
                Self::Cuesheet(_) => BlockType::Cuesheet,
                Self::Picture(_) => BlockType::Picture,
            }
        }

        pub fn optional_block_type(&self) -> OptionalBlockType {
            match self {
                Self::Padding(_) => OptionalBlockType::Padding,
                Self::Application(_) => OptionalBlockType::Application,
                Self::SeekTable(_) => OptionalBlockType::SeekTable,
                Self::VorbisComment(_) => OptionalBlockType::VorbisComment,
                Self::Cuesheet(_) => OptionalBlockType::Cuesheet,
                Self::Picture(_) => OptionalBlockType::Picture,
            }
        }
    }

    impl From<OptionalBlock> for Block {
        fn from(block: OptionalBlock) -> Block {
            match block {
                OptionalBlock::Padding(p) => Block::Padding(p),
                OptionalBlock::Application(a) => Block::Application(a),
                OptionalBlock::SeekTable(s) => Block::SeekTable(s),
                OptionalBlock::VorbisComment(v) => Block::VorbisComment(v),
                OptionalBlock::Cuesheet(c) => Block::Cuesheet(c),
                OptionalBlock::Picture(p) => Block::Picture(p),
            }
        }
    }

    impl TryFrom<Block> for OptionalBlock {
        type Error = Streaminfo;

        fn try_from(block: Block) -> Result<Self, Streaminfo> {
            match block {
                Block::Streaminfo(s) => Err(s),
                Block::Padding(p) => Ok(OptionalBlock::Padding(p)),
                Block::Application(a) => Ok(OptionalBlock::Application(a)),
                Block::SeekTable(s) => Ok(OptionalBlock::SeekTable(s)),
                Block::VorbisComment(v) => Ok(OptionalBlock::VorbisComment(v)),
                Block::Cuesheet(c) => Ok(OptionalBlock::Cuesheet(c)),
                Block::Picture(p) => Ok(OptionalBlock::Picture(p)),
            }
        }
    }

    impl AsBlockRef for OptionalBlock {
        fn as_block_ref(&self) -> BlockRef<'_> {
            match self {
                Self::Padding(p) => BlockRef::Padding(p),
                Self::Application(a) => BlockRef::Application(a),
                Self::SeekTable(s) => BlockRef::SeekTable(s),
                Self::VorbisComment(v) => BlockRef::VorbisComment(v),
                Self::Cuesheet(v) => BlockRef::Cuesheet(v),
                Self::Picture(p) => BlockRef::Picture(p),
            }
        }
    }

    pub trait OptionalMetadataBlock: Into<OptionalBlock> + TryFrom<OptionalBlock> {
        fn try_from_opt_block(block: &OptionalBlock) -> Result<&Self, &OptionalBlock>;

        fn try_from_opt_block_mut(
            block: &mut OptionalBlock,
        ) -> Result<&mut Self, &mut OptionalBlock>;
    }
}

/// The channel mask
///
/// This field is used to communicate that the channels
/// in the file differ from FLAC's default channel assignment
/// definitions.
///
/// It is generally used for multi-channel audio
/// and stored within the [`VorbisComment`] metadata block
/// as the [`fields::CHANNEL_MASK`] field.
///
/// # Example
///
/// ```
/// use flac_codec::metadata::{ChannelMask, Channel};
///
/// let mask = "0x003F".parse::<ChannelMask>().unwrap();
///
/// let mut channels = mask.channels();
/// assert_eq!(channels.next(), Some(Channel::FrontLeft));
/// assert_eq!(channels.next(), Some(Channel::FrontRight));
/// assert_eq!(channels.next(), Some(Channel::FrontCenter));
/// assert_eq!(channels.next(), Some(Channel::Lfe));
/// assert_eq!(channels.next(), Some(Channel::BackLeft));
/// assert_eq!(channels.next(), Some(Channel::BackRight));
/// assert_eq!(channels.next(), None);
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct ChannelMask {
    mask: u32,
}

impl ChannelMask {
    /// Iterates over all the mask's defined channels
    pub fn channels(&self) -> impl Iterator<Item = Channel> {
        [
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::FrontCenter,
            Channel::Lfe,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::FrontLeftOfCenter,
            Channel::FrontRightOfCenter,
            Channel::BackCenter,
            Channel::SideLeft,
            Channel::SideRight,
            Channel::TopCenter,
            Channel::TopFrontLeft,
            Channel::TopFrontCenter,
            Channel::TopFrontRight,
            Channel::TopRearLeft,
            Channel::TopRearCenter,
            Channel::TopRearRight,
        ]
        .into_iter()
        .filter(|channel| (*channel as u32 & self.mask) != 0)
    }

    fn from_channels(channels: u8) -> Self {
        match channels {
            1 => Self {
                mask: Channel::FrontCenter as u32,
            },
            2 => Self {
                mask: Channel::FrontLeft as u32 | Channel::FrontRight as u32,
            },
            3 => Self {
                mask: Channel::FrontLeft as u32
                    | Channel::FrontRight as u32
                    | Channel::FrontCenter as u32,
            },
            4 => Self {
                mask: Channel::FrontLeft as u32
                    | Channel::FrontRight as u32
                    | Channel::BackLeft as u32
                    | Channel::BackRight as u32,
            },
            5 => Self {
                mask: Channel::FrontLeft as u32
                    | Channel::FrontRight as u32
                    | Channel::FrontCenter as u32
                    | Channel::SideLeft as u32
                    | Channel::SideRight as u32,
            },
            6 => Self {
                mask: Channel::FrontLeft as u32
                    | Channel::FrontRight as u32
                    | Channel::FrontCenter as u32
                    | Channel::Lfe as u32
                    | Channel::SideLeft as u32
                    | Channel::SideRight as u32,
            },
            7 => Self {
                mask: Channel::FrontLeft as u32
                    | Channel::FrontRight as u32
                    | Channel::FrontCenter as u32
                    | Channel::Lfe as u32
                    | Channel::BackCenter as u32
                    | Channel::SideLeft as u32
                    | Channel::SideRight as u32,
            },
            8 => Self {
                mask: Channel::FrontLeft as u32
                    | Channel::FrontRight as u32
                    | Channel::FrontCenter as u32
                    | Channel::Lfe as u32
                    | Channel::BackLeft as u32
                    | Channel::BackRight as u32
                    | Channel::SideLeft as u32
                    | Channel::SideRight as u32,
            },
            // FLAC files are limited to 1-8 channels
            _ => panic!("undefined channel count"),
        }
    }
}

impl std::str::FromStr for ChannelMask {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('x').ok_or(())? {
            ("0", hex) => u32::from_str_radix(hex, 16)
                .map(|mask| ChannelMask { mask })
                .map_err(|_| ()),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for ChannelMask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "0x{:04x}", self.mask)
    }
}

impl From<ChannelMask> for u32 {
    fn from(mask: ChannelMask) -> u32 {
        mask.mask
    }
}

impl From<u32> for ChannelMask {
    fn from(mask: u32) -> ChannelMask {
        ChannelMask { mask }
    }
}

/// An individual channel mask channel
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Channel {
    /// Front left channel
    FrontLeft = 0b1,

    /// Front right channel
    FrontRight = 0b10,

    /// Front center channel
    FrontCenter = 0b100,

    /// Low-frequency effects (LFE) channel
    Lfe = 0b1000,

    /// Back left channel
    BackLeft = 0b10000,

    /// Back right channel
    BackRight = 0b100000,

    /// Front left of center channel
    FrontLeftOfCenter = 0b1000000,

    /// Front right of center channel
    FrontRightOfCenter = 0b10000000,

    /// Back center channel
    BackCenter = 0b100000000,

    /// Side left channel
    SideLeft = 0b1000000000,

    /// Side right channel
    SideRight = 0b10000000000,

    /// Top center channel
    TopCenter = 0b100000000000,

    /// Top front left channel
    TopFrontLeft = 0b1000000000000,

    /// Top front center channel
    TopFrontCenter = 0b10000000000000,

    /// Top front right channel
    TopFrontRight = 0b100000000000000,

    /// Top rear left channel
    TopRearLeft = 0b1000000000000000,

    /// Top rear center channel
    TopRearCenter = 0b10000000000000000,

    /// Top rear right channel
    TopRearRight = 0b100000000000000000,
}

impl std::fmt::Display for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::FrontLeft => "front left".fmt(f),
            Self::FrontRight => "front right".fmt(f),
            Self::FrontCenter => "front center".fmt(f),
            Self::Lfe => "LFE".fmt(f),
            Self::BackLeft => "back left".fmt(f),
            Self::BackRight => "back right".fmt(f),
            Self::FrontLeftOfCenter => "front left of center".fmt(f),
            Self::FrontRightOfCenter => "front right of center".fmt(f),
            Self::BackCenter => "back center".fmt(f),
            Self::SideLeft => "side left".fmt(f),
            Self::SideRight => "side right".fmt(f),
            Self::TopCenter => "top center".fmt(f),
            Self::TopFrontLeft => "top front left".fmt(f),
            Self::TopFrontCenter => "top front center".fmt(f),
            Self::TopFrontRight => "top front right".fmt(f),
            Self::TopRearLeft => "top rear left".fmt(f),
            Self::TopRearCenter => "top rear center".fmt(f),
            Self::TopRearRight => "top rear right".fmt(f),
        }
    }
}
