// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For handling a FLAC file's metadata blocks

use crate::Error;
use bitstream_io::{
    BigEndian, BitRead, BitReader, BitWrite, ByteRead, ByteReader, FromBitStream,
    FromBitStreamUsing, FromBitStreamWith, LittleEndian, SignedBitCount, ToBitStream,
    ToBitStreamUsing,
};
use std::num::NonZero;
use std::path::Path;

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
///         last: true,
///         block_type: BlockType::Streaminfo,
///         size: 0x00_00_22u16.into(),
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

/// A type of FLAC metadata block
pub trait MetadataBlock: ToBitStream<Error: Into<Error>> + Into<Block> {
    /// The metadata block's type
    const TYPE: BlockType;

    /// Whether the block can occur multiple times in a file
    const MULTIPLE: bool;
}

impl BlockHeader {
    fn new<M: MetadataBlock>(last: bool, block: &M) -> Result<Self, Error> {
        use bitstream_io::write::Overflowed;

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
                self.0 % 8 == 0
            }
        }

        impl From<BlockBits> for BlockSize {
            fn from(BlockBits(u): BlockBits) -> Self {
                assert!(u % 8 == 0);
                Self(u / 8)
            }
        }

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

/// A 24-bit block size value, with safeguards against overflow
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct BlockSize(u32);

impl BlockSize {
    const MAX: u32 = (1 << 24) - 1;

    /// Our current value as a u32
    fn get(&self) -> u32 {
        self.0
    }
}

impl BlockSize {
    fn checked_add(self, rhs: Self) -> Option<Self> {
        self.0
            .checked_add(rhs.0)
            .filter(|s| *s <= Self::MAX)
            .map(Self)
    }

    fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
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
pub struct BlockReader<R: std::io::Read> {
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

impl<R: std::io::Read> BlockReader<R> {
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

impl<R: std::io::Read> Iterator for BlockReader<R> {
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
                    b"fLaC" => {
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
/// Because this may perform many small reads,
/// using a buffered reader may greatly improve performance
/// when reading from a raw `File`.
pub fn read_blocks<R: std::io::Read>(r: R) -> BlockReader<R> {
    BlockReader::new(r)
}

/// Writes iterator of blocks to the given writer.
///
/// Because this may perform many small writes,
/// buffering writes may greatly improve performance
/// when writing to a raw `File`.
///
/// # Errors
///
/// Passes along any I/O errors from the underlying stream.
/// May also generate an error if any of the blocks are invalid
/// (e.g. STREAMINFO not being the first block, any block is too large, etc.).
pub fn write_blocks<B: AsBlockRef>(
    blocks: impl IntoIterator<Item = B>,
    mut w: impl std::io::Write,
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
    w.write_all(b"fLaC").map_err(Error::Io)?;

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

/// Whether to perform or rollback metadata changes
pub enum Save {
    /// Commit changes to disk, if possible
    Commit,
    /// Abort changes
    Rollback,
}

/// Given a Path, attempts to update FLAC metadata blocks
///
/// # Errors
///
/// Returns error if unable to read metadata blocks,
/// unable to write blocks, or if the existing or updated
/// blocks do not conform to the FLAC file specification.
pub fn update_file<P, E>(
    path: P,
    f: impl FnOnce(&mut BlockList) -> Result<Save, E>,
) -> Result<(), E>
where
    P: AsRef<Path>,
    E: From<Error>,
{
    use std::fs::OpenOptions;

    update(
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
/// Applies closure `f` to the blocks and attempts to update
/// them if `Save::Commit` is returned.
///
/// If the updated blocks can be made the same size as the
/// original file by adjusting padding, the file will be
/// completely overwritten with new contents.
///
/// If the new blocks are too large (or small) to fit into
/// the original file, the original file is dropped
/// and the `rebuilt` closure is called to build a new
/// file.  The file's contents are then dumped into the new file.
pub fn update<F, N, E>(
    mut original: F,
    rebuilt: impl FnOnce() -> std::io::Result<N>,
    f: impl FnOnce(&mut BlockList) -> Result<Save, E>,
) -> Result<(), E>
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
        write_blocks(blocks, &mut tmp)?;
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

    let mut reader = Counter::new(BufReader::new(&mut original));

    let mut blocks =
        read_blocks(Read::by_ref(&mut reader)).collect::<Result<Result<BlockList, _>, _>>()??;

    let Counter {
        stream: reader,
        count: old_size,
    } = reader;

    if matches!(f(&mut blocks)?, Save::Rollback) {
        return Ok(());
    }

    let new_size = {
        let mut new_size = Counter::new(sink());
        write_blocks(blocks.blocks(), &mut new_size)?;
        new_size.count
    };

    match new_size.cmp(&old_size) {
        Ordering::Less => {
            // blocks have shrunk in size, so try to expand
            // PADDING block to hold additional bytes
            match grow_padding(&mut blocks, old_size - new_size) {
                Ok(()) => {
                    original.rewind().map_err(Error::Io)?;
                    write_blocks(blocks, BufWriter::new(original))?
                }
                Err(()) => rebuild_file(rebuilt, reader, blocks)?,
            }
            Ok(())
        }
        Ordering::Equal => {
            // blocks are the same size, so no need to adjust padding
            original.rewind().map_err(Error::Io)?;
            Ok(write_blocks(blocks, BufWriter::new(original))?)
        }
        Ordering::Greater => {
            // blocks have grown in size, so try to shrink
            // PADDING block to hold additional bytes
            match shrink_padding(&mut blocks, new_size - old_size) {
                Ok(()) => {
                    original.rewind().map_err(Error::Io)?;
                    write_blocks(blocks.into_iter(), BufWriter::new(original))?
                }
                Err(()) => rebuild_file(rebuilt, reader, blocks)?,
            }
            Ok(())
        }
    }
}

/// Any possible FLAC metadata block
///
/// Each block consists of a [`BlockHeader`] followed by the block's contents.
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
#[derive(Debug, Copy, Clone)]
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

        impl AsBlockRef for $t {
            fn as_block_ref(&self) -> BlockRef<'_> {
                BlockRef::$v(self)
            }
        }
    };
}

macro_rules! optional_block {
    ($t:ty, $v:ident) => {
        impl OptionalMetadataBlock for $t {}

        impl private::OptionalMetadataBlock for $t {
            fn try_from_opt_block(block: &private::OptionalBlock) -> Option<&Self> {
                match block {
                    private::OptionalBlock::$v(comment) => Some(comment),
                    _ => None,
                }
            }

            fn try_from_opt_block_mut(block: &mut private::OptionalBlock) -> Option<&mut Self> {
                match block {
                    private::OptionalBlock::$v(comment) => Some(comment),
                    _ => None,
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
/// This block must *always* be present in a FLAC file,
/// must *always* be the first metadata block in the stream,
/// and must not be present more than once.
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
}

block!(Streaminfo, Streaminfo, false);

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
/// This block may occur multiple times in a FLAC file.
///
/// The contents of a PADDING block are all 0 bytes.
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
#[derive(Debug, Clone, Eq, PartialEq)]
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
/// assert_eq!(
///     r.parse_using::<SeekTable>(header.size).unwrap(),
///     SeekTable {
///         points: vec![
///             SeekPoint {
///                 sample_offset: Some(0x00),
///                 byte_offset: 0x00,
///                 frame_samples: 0x14,
///             },
///             SeekPoint {
///                 sample_offset: Some(0x14),
///                 byte_offset: 0x0c,
///                 frame_samples: 0x14,
///             },
///             SeekPoint {
///                 sample_offset: Some(0x28),
///                 byte_offset: 0x22,
///                 frame_samples: 0x14,
///             },
///             SeekPoint {
///                 sample_offset: Some(0x3c),
///                 byte_offset: 0x3c,
///                 frame_samples: 0x14,
///             },
///         ],
///     },
/// );
///
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SeekTable {
    /// The seek table's individual seek points
    pub points: Vec<SeekPoint>,
}

block!(SeekTable, SeekTable, false);
optional_block!(SeekTable, SeekTable);

impl FromBitStreamUsing for SeekTable {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: BlockSize) -> Result<Self, Self::Error> {
        match (size.get() / 18, size.get() % 18) {
            (p, 0) => {
                let mut points = Vec::with_capacity(p.try_into().unwrap());

                for _ in 0..p {
                    let point: SeekPoint = r.parse()?;
                    match point.sample_offset {
                        None => points.push(point),
                        Some(our_offset) => match points.last() {
                            Some(SeekPoint {
                                sample_offset: Some(last_offset),
                                ..
                            }) if our_offset <= *last_offset => {
                                return Err(Error::InvalidSeekTablePoint);
                            }
                            _ => points.push(point),
                        },
                    }
                }

                Ok(Self { points })
            }
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
                    last_offset = point.sample_offset;
                    w.build(point).map_err(Error::Io)
                }
                Some(last_offset) => match point {
                    SeekPoint {
                        sample_offset: Some(our_offset),
                        ..
                    } => match our_offset > last_offset {
                        true => {
                            *last_offset = *our_offset;
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
pub struct SeekPoint {
    /// The sample number of the first sample in the target frame,
    /// or `None` for placeholder points
    pub sample_offset: Option<u64>,
    /// Offset, in bytes, from the first byte of the first frame header
    /// to the first byte in the target frame's header
    pub byte_offset: u64,
    /// Number of samples in the target frame
    pub frame_samples: u16,
}

impl FromBitStream for SeekPoint {
    type Error = std::io::Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        Ok(Self {
            sample_offset: r.read_to().map(|o| (o != u64::MAX).then_some(o))?,
            byte_offset: r.read_to()?,
            frame_samples: r.read_to()?,
        })
    }
}

impl ToBitStream for SeekPoint {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(self.sample_offset.unwrap_or(u64::MAX))?;
        w.write_from(self.byte_offset)?;
        w.write_from(self.frame_samples)
    }
}

/// A VORBIS_COMMENT metadata block
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
/// assert_eq!(
///     r.parse::<VorbisComment>().unwrap(),
///     VorbisComment {
///         vendor_string: "reference libFLAC 1.4.3 20230623".to_string(),
///         fields: vec![
///              "TITLE=Testing".to_string(),
///              "ALBUM=Test Album".to_string(),
///         ],
///     },
/// );
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
    /// Name of current work
    pub const TITLE: &str = "TITLE";

    /// Name of the artist generally responsible for the current work
    pub const ARTIST: &str = "ARTIST";

    /// Name of the collection the current work belongs to
    pub const ALBUM: &str = "ALBUM";

    /// The channel mask of multi-channel audio streams
    pub const CHANNEL_MASK: &str = "WAVEFORMATEXTENSIBLE_CHANNEL_MASK";

    /// Given a field name, returns first matching value, if any
    ///
    /// Fields are matched case-insensitively
    pub fn field(&self, field: &str) -> Option<&str> {
        self.field_values(field).next()
    }

    /// Given a field name, iterates over any matching values
    ///
    /// Fields are matched case-insensitively
    pub fn field_values(&self, field: &str) -> impl Iterator<Item = &str> {
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
    pub fn append_field<S>(&mut self, field: &str, value: S)
    where
        S: std::fmt::Display,
    {
        assert!(!field.contains('='), "field must not contain '='");

        self.fields.push(format!("{field}={value}"));
    }

    /// Removes any matching instances of the given field
    ///
    /// Fields are matched case-insensitively
    pub fn remove_field(&mut self, field: &str) {
        self.fields.retain(|f| match f.split_once('=') {
            Some((key, _)) => !key.eq_ignore_ascii_case(field),
            None => true,
        });
    }

    /// Replaces any instances of the given field with value
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Panics
    ///
    /// Panics if field contains the `=` character.
    pub fn set_field_value<S>(&mut self, field: &str, value: S)
    where
        S: std::fmt::Display,
    {
        self.remove_field(field);
        self.append_field(field, value);
    }

    /// Replaces any instances of the given field with the given values
    ///
    /// Fields are matched case-insensitively
    ///
    /// # Panics
    ///
    /// Panics if field contains the `=` character
    pub fn set_field_values<S, I>(&mut self, field: &str, values: I)
    where
        S: std::fmt::Display,
        I: IntoIterator<Item = S>,
    {
        assert!(!field.contains('='), "field must not contain '='");

        self.remove_field(field);
        self.fields
            .extend(values.into_iter().map(|value| format!("{field}={value}")));
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

/// A CUESHEET metadata block
///
/// This block may occur multiple times in a FLAC file.
///
/// | Bits  | Field | Meaning |
/// |------:|------:|---------|
/// | 128×8 | `catalog_number` | media catalog number, in ASCII
/// | 64 | `lead_in_samples` | number of lead-in samples
/// | 1  | `is_cdda` | whether cuesheet corresponds to CD-DA
/// | 7+258×8 | padding | all 0 bits
/// | | `tracks` | cuesheet track₀, cuesheet track₁, …
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Cuesheet {
    /// Media catalog number in ASCII printable characters
    pub catalog_number: Box<[u8; 128]>,
    /// Number of lead-in samples
    pub lead_in_samples: u64,
    /// Whether cuesheet corresponds to CA-DA
    pub is_cdda: bool,
    /// Cuesheet's tracks
    pub tracks: Vec<CuesheetTrack>,
}

block!(Cuesheet, Cuesheet, true);
optional_block!(Cuesheet, Cuesheet);

impl FromBitStream for Cuesheet {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let catalog_number = r.read_to()?;
        let lead_in_samples = r.read_to()?;
        let is_cdda = r.read_bit()?;
        r.skip(7 + 258 * 8)?;

        Ok(Self {
            catalog_number: Box::new(catalog_number),
            lead_in_samples,
            is_cdda,
            tracks: (0..r.read_to::<u8>()?)
                .map(|_| r.parse_using::<CuesheetTrack>(is_cdda))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl ToBitStream for Cuesheet {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(*self.catalog_number)?;
        w.write_from(self.lead_in_samples)?;
        w.write_bit(self.is_cdda)?;
        w.pad(7 + 258 * 8)?;
        w.write_from(u8::try_from(self.tracks.len()).map_err(|_| Error::ExcessiveCuesheetTracks)?)?;
        self.tracks
            .iter()
            .try_for_each(|track| w.build_using(track, self.is_cdda))
    }
}

/// An individual CUESHEET track
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 64   | `offset` | offset of first index point, in samples
/// | 8    | `number` | track number
/// | 12×8 | `isrc`   | track ISRC
/// | 1    | `non_audio`| whether track is non-audio
/// | 1    | `pre_emphasis` | whether track has pre-emphasis
/// | 6+13×8 | padding | all 0 bits
/// | 8    | point count | number index points
/// |      | | index point₀, index point₁, …
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CuesheetTrack {
    /// Offset of the first index point in samples,
    /// relative to the beginning of the FLAC audio stream
    pub offset: u64,
    /// Track number
    pub number: u8,
    /// Track ISRC
    pub isrc: Option<[u8; 12]>,
    /// Whether track is non-audio
    pub non_audio: bool,
    /// Whether track has pre-emphasis
    pub pre_emphasis: bool,
    /// The tracks' index points
    pub index_points: Vec<CuesheetIndexPoint>,
}

impl FromBitStreamUsing for CuesheetTrack {
    type Error = Error;
    type Context = bool;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, is_cdda: bool) -> Result<Self, Self::Error> {
        let offset: u64 = r.read_to()?;
        if is_cdda && offset % 588 != 0 {
            // CDDA tracks must have a sample count divisible by 588
            return Err(Error::InvalidCuesheetOffset);
        }
        let number = r.read_to()?;
        let isrc = r
            .read_to()
            .map(|isrc: [u8; 12]| (!isrc.iter().all(|c| *c == 0)).then_some(isrc))?;
        let [non_audio, pre_emphasis] = r.read::<1, _>()?;
        r.skip(6 + 13 * 8)?;
        let point_count = r.read_to::<u8>()?;

        match TrackType::new(is_cdda, number) {
            TrackType::Regular => {
                if point_count == 0 {
                    return Err(Error::InvalidCuesheetIndexPoints);
                }
            }
            TrackType::LeadOut => {
                if point_count != 0 {
                    return Err(Error::InvalidCuesheetIndexPoints);
                }
            }
            TrackType::Invalid => {
                return Err(Error::InvalidCuesheetTrackNumber);
            }
        }

        let mut index_points = Vec::with_capacity(point_count.into());
        for _ in 0..point_count {
            let point: CuesheetIndexPoint = r.parse_using(is_cdda)?;
            match index_points.last() {
                // first index point must have a number of 0 or 1
                None => match point.number {
                    0 | 1 => index_points.push(point),
                    _ => return Err(Error::InvalidCuesheetIndexPointNum),
                },
                // subsequent index points must increment by 1
                Some(CuesheetIndexPoint { number, .. }) => match point.number == number + 1 {
                    true => index_points.push(point),
                    false => return Err(Error::InvalidCuesheetIndexPointNum),
                },
            }
        }

        Ok(Self {
            offset,
            number,
            isrc,
            non_audio,
            pre_emphasis,
            index_points,
        })
    }
}

impl ToBitStreamUsing for CuesheetTrack {
    type Error = Error;
    type Context = bool;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W, is_cdda: bool) -> Result<(), Self::Error> {
        if is_cdda && self.offset % 588 != 0 {
            return Err(Error::InvalidCuesheetOffset);
        }
        w.write_from(self.offset)?;
        w.write_from(self.number)?;
        w.write_from(self.isrc.unwrap_or([0; 12]))?;
        w.write_bit(self.non_audio)?;
        w.write_bit(self.pre_emphasis)?;
        w.pad(6 + 13 * 8)?;
        let point_count = u8::try_from(self.index_points.len())
            .map_err(|_| Error::ExcessiveCuesheetIndexPoints)?;
        w.write_from(point_count)?;

        match TrackType::new(is_cdda, self.number) {
            TrackType::Regular => match self.index_points.as_slice() {
                [] => Err(Error::InvalidCuesheetIndexPoints),
                index_points => {
                    let mut last_point = None;
                    index_points
                        .iter()
                        .try_for_each(|point| match last_point.as_mut() {
                            // first index point must have a number of 0 or 1
                            None => match point.number {
                                0 | 1 => {
                                    last_point = Some(point.number);
                                    w.build_using(point, is_cdda)
                                }
                                _ => Err(Error::InvalidCuesheetIndexPointNum),
                            },
                            Some(previous_number) => match point.number == *previous_number + 1 {
                                true => {
                                    *previous_number = point.number;
                                    w.build_using(point, is_cdda)
                                }
                                false => Err(Error::InvalidCuesheetIndexPointNum),
                            },
                        })
                }
            },
            TrackType::LeadOut => match self.index_points.as_slice() {
                [] => Ok(()),
                _ => Err(Error::InvalidCuesheetIndexPoints),
            },
            TrackType::Invalid => Err(Error::InvalidCuesheetTrackNumber),
        }
    }
}

enum TrackType {
    Regular,
    LeadOut,
    Invalid,
}

impl TrackType {
    fn new(is_cdda: bool, track_number: u8) -> Self {
        match is_cdda {
            true => match track_number {
                1..=99 => Self::Regular,
                170 => Self::LeadOut,
                _ => Self::Invalid,
            },
            false => match track_number {
                255 => Self::LeadOut,
                _ => Self::Regular,
            },
        }
    }
}

/// An individual CUESHEET track index point
///
/// | Bits | Field | Meaning |
/// |-----:|------:|---------|
/// | 64   | `offset` | index point offset, in samples
/// | 8    | `number` | index point number
/// | 3×8  | padding  | all 0 bits
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CuesheetIndexPoint {
    /// Offset in samples
    pub offset: u64,
    /// Track index point number
    pub number: u8,
}

impl FromBitStreamUsing for CuesheetIndexPoint {
    type Error = Error;
    type Context = bool;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, is_cdda: bool) -> Result<Self, Self::Error> {
        let offset: u64 = r.read_to()?;
        if is_cdda && offset % 588 != 0 {
            return Err(Error::InvalidCuesheetOffset);
        }
        let number = r.read_to()?;
        r.skip(3 * 8)?;
        Ok(Self { offset, number })
    }
}

impl ToBitStreamUsing for CuesheetIndexPoint {
    type Error = Error;
    type Context = bool;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W, is_cdda: bool) -> Result<(), Self::Error> {
        match is_cdda && self.offset % 588 != 0 {
            false => {
                w.write_from(self.offset)?;
                w.write_from(self.number)?;
                w.pad(3 * 8)?;
                Ok(())
            }
            true => Err(Error::InvalidCuesheetOffset),
        }
    }
}

/// A PICTURE metadata block
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
#[derive(Debug, Clone, Eq, PartialEq)]
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
            let mut v = self
                .blocks
                .extract_if(.., |b| b.block_type() == B::TYPE)
                .collect::<Vec<_>>();

            self.blocks.push(block.into());

            v.pop().and_then(|b| b.try_into().ok())
        }
    }

    /// Gets reference to metadata block, if present
    ///
    /// If the block type occurs multiple times,
    /// this returns the first instance.
    pub fn get<B: OptionalMetadataBlock>(&self) -> Option<&B> {
        self.blocks.iter().find_map(B::try_from_opt_block)
    }

    /// Gets mutable reference to metadata block, if present
    ///
    /// If the block type occurs multiple times,
    /// this returns the first instance.
    pub fn get_mut<B: OptionalMetadataBlock>(&mut self) -> Option<&mut B> {
        self.blocks.iter_mut().find_map(B::try_from_opt_block_mut)
    }

    /// Gets references to all metadata blocks of the given type
    pub fn get_all<'b, B: OptionalMetadataBlock + 'b>(&'b self) -> impl Iterator<Item = &'b B> {
        self.blocks.iter().filter_map(B::try_from_opt_block)
    }

    /// Gets exclusive references to all metadata blocks of the given type
    pub fn get_all_mut<'b, B: OptionalMetadataBlock + 'b>(
        &'b mut self,
    ) -> impl Iterator<Item = &'b mut B> {
        self.blocks.iter_mut().filter_map(B::try_from_opt_block_mut)
    }

    /// Removes all instances of the given metadata block type
    pub fn remove<B: OptionalMetadataBlock>(&mut self) {
        self.blocks.retain(|b| b.block_type() != B::TYPE)
    }

    /// Updates Vorbis comment, creating a new block if necessary
    pub fn update_comment(&mut self, f: impl FnOnce(&mut VorbisComment)) {
        match self.get_mut() {
            Some(comment) => f(comment),
            None => {
                let mut c = VorbisComment::default();
                f(&mut c);
                self.blocks.push(c.into());
            }
        }
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

/// A type of FLAC metadata block which is not required
///
/// The STREAMINFO block is required.  All others are optional.
pub trait OptionalMetadataBlock: MetadataBlock + private::OptionalMetadataBlock {}

mod private {
    use super::{
        Application, AsBlockRef, Block, BlockRef, BlockType, Cuesheet, Padding, Picture, SeekTable,
        Streaminfo, VorbisComment,
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
        fn try_from_opt_block(block: &OptionalBlock) -> Option<&Self>;

        fn try_from_opt_block_mut(block: &mut OptionalBlock) -> Option<&mut Self>;
    }
}
