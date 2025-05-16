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
    FromBitStreamWith, LittleEndian, SignedBitCount, ToBitStream, ToBitStreamWith,
};
use std::num::NonZero;
use std::path::Path;

/// A FLAC metadata block header
#[derive(Debug)]
pub struct BlockHeader {
    /// Whether we are the final block
    pub last: bool,
    /// Our block type
    pub block_type: BlockType,
    /// Our block size, in bytes
    pub size: BlockSize,
}

/// A type of FLAC metadata block
pub trait MetadataBlock: ToBitStream<Error: Into<Error>> {
    /// The metadata block's type
    const TYPE: BlockType;
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
    Streaminfo,
    /// The PADDING block
    Padding,
    /// The APPLICATION block
    Application,
    /// The SEEKTABLE block
    SeekTable,
    /// The VORBIS_COMMENT block
    VorbisComment,
    /// The CUESHEET block
    Cuesheet,
    /// The PICTURE block
    Picture,
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
#[derive(Debug, Default, Copy, Clone)]
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

impl TryFrom<usize> for BlockSize {
    type Error = BlockSizeOverflow;

    fn try_from(u: usize) -> Result<Self, Self::Error> {
        u32::try_from(u)
            .map_err(|_| BlockSizeOverflow)
            .and_then(|s| (s <= Self::MAX).then_some(Self(s)).ok_or(BlockSizeOverflow))
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
        Some((last, streaminfo @ BlockRef::Streaminfo(_))) => w.build_with(&streaminfo, last)?,
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
                w.build_with(&vorbiscomment.as_block_ref(), &last)
            }
            true => Err(Error::MultipleVorbisComment),
        },
        seektable @ BlockRef::SeekTable(_) => match seektable_read {
            false => {
                seektable_read = true;
                w.build_with(&seektable.as_block_ref(), &last)
            }
            true => Err(Error::MultipleSeekTable),
        },
        picture @ BlockRef::Picture(Picture {
            picture_type: PictureType::Png32x32,
            ..
        }) => {
            if !png_read {
                png_read = true;
                w.build_with(&picture.as_block_ref(), &last)
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
                w.build_with(&picture.as_block_ref(), &last)
            } else {
                Err(Error::MultipleGeneralIcon)
            }
        }
        block => w.build_with(&block.as_block_ref(), &last),
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
pub fn update<P, E>(path: P, f: impl FnOnce(&mut Vec<Block>) -> Result<Save, E>) -> Result<(), E>
where
    P: AsRef<Path>,
    E: From<Error>,
{
    use crate::Counter;
    use std::cmp::Ordering;
    use std::fs::OpenOptions;
    use std::io::{BufReader, BufWriter, Read, Seek, Write, sink};

    fn rebuild_file<P, R>(p: P, mut r: R, blocks: Vec<Block>) -> Result<(), Error>
    where
        P: AsRef<Path>,
        R: Read,
    {
        // dump our new blocks and remaining FLAC data to temp file
        let mut tmp = Vec::new();
        write_blocks(&blocks, &mut tmp)?;
        std::io::copy(&mut r, &mut tmp).map_err(Error::Io)?;
        drop(r);

        // open original file and rewrite it with tmp file contents
        std::fs::File::create(p)
            .and_then(|mut f| f.write_all(tmp.as_slice()))
            .map_err(Error::Io)
    }

    /// Returns Ok if successful
    fn grow_padding(blocks: &mut [Block], more_bytes: u64) -> Result<(), ()> {
        // if a block set has more than one PADDING, we'll try the first
        // rather than attempt to grow each in turn
        //
        // this is the most common case
        let padding = blocks
            .iter_mut()
            .find_map(|b| match b {
                Block::Padding(p) => Some(p),
                _ => None,
            })
            .ok_or(())?;

        padding.size = padding
            .size
            .checked_add(more_bytes.try_into().map_err(|_| ())?)
            .ok_or(())?;

        Ok(())
    }

    /// Returns Ok if successful
    fn shrink_padding(blocks: &mut [Block], fewer_bytes: u64) -> Result<(), ()> {
        // if a block set has more than one PADDING, we'll try the first
        // rather than attempt to grow each in turn
        //
        // this is the most common case
        let padding = blocks
            .iter_mut()
            .find_map(|b| match b {
                Block::Padding(p) => Some(p),
                _ => None,
            })
            .ok_or(())?;

        padding.size = padding
            .size
            .checked_sub(fewer_bytes.try_into().map_err(|_| ())?)
            .ok_or(())?;

        Ok(())
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .truncate(false)
        .create(false)
        .open(path.as_ref())
        .map_err(Error::Io)?;

    let mut reader = Counter::new(BufReader::new(Read::by_ref(&mut file)));

    let mut blocks = read_blocks(Read::by_ref(&mut reader)).collect::<Result<Vec<_>, _>>()?;

    let Counter {
        stream: reader,
        count: old_size,
    } = reader;

    if matches!(f(&mut blocks)?, Save::Rollback) {
        return Ok(());
    }

    let new_size = {
        let mut new_size = Counter::new(sink());
        write_blocks(&blocks, &mut new_size)?;
        new_size.count
    };

    match new_size.cmp(&old_size) {
        Ordering::Less => {
            // blocks have shrunk in size, so try to expand
            // PADDING block to hold additional bytes
            match grow_padding(&mut blocks, old_size - new_size) {
                Ok(()) => {
                    file.rewind().map_err(Error::Io)?;
                    write_blocks(&blocks, BufWriter::new(file))?
                }
                Err(()) => rebuild_file(path, reader, blocks)?,
            }
            Ok(())
        }
        Ordering::Equal => {
            // blocks are the same size, so no need to adjust padding
            file.rewind().map_err(Error::Io)?;
            Ok(write_blocks(&blocks, BufWriter::new(file))?)
        }
        Ordering::Greater => {
            // blocks have grown in size, so try to shrink
            // PADDING block to hold additional bytes
            match shrink_padding(&mut blocks, new_size - old_size) {
                Ok(()) => {
                    file.rewind().map_err(Error::Io)?;
                    write_blocks(&blocks, BufWriter::new(file))?
                }
                Err(()) => rebuild_file(path, reader, blocks)?,
            }
            Ok(())
        }
    }
}

/// Any possible FLAC metadata block
#[derive(Debug, Clone)]
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

macro_rules! to_block {
    ($t:ty, $b:ident) => {
        impl From<$t> for Block {
            fn from(b: $t) -> Self {
                Self::$b(b)
            }
        }
    };
}

to_block!(Streaminfo, Streaminfo);
to_block!(Padding, Padding);
to_block!(Application, Application);
to_block!(SeekTable, SeekTable);
to_block!(VorbisComment, VorbisComment);
to_block!(Cuesheet, Cuesheet);
to_block!(Picture, Picture);

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
            BlockType::Padding => Ok(Block::Padding(r.parse_with(&header.size)?)),
            BlockType::Application => Ok(Block::Application(r.parse_with(&header.size)?)),
            BlockType::SeekTable => Ok(Block::SeekTable(r.parse_with(&header.size)?)),
            BlockType::VorbisComment => Ok(Block::VorbisComment(r.parse()?)),
            BlockType::Cuesheet => Ok(Block::Cuesheet(r.parse()?)),
            BlockType::Picture => Ok(Block::Picture(r.parse()?)),
        }
    }
}

impl ToBitStreamWith<'_> for Block {
    type Context = bool;
    type Error = Error;

    // builds to writer with header
    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W, is_last: &bool) -> Result<(), Error> {
        match self {
            Self::Streaminfo(streaminfo) => w
                .build(&BlockHeader::new(*is_last, streaminfo)?)
                .and_then(|()| w.build(streaminfo).map_err(Error::Io)),
            Self::Padding(padding) => w
                .build(&BlockHeader::new(*is_last, padding)?)
                .and_then(|()| w.build(padding).map_err(Error::Io)),
            Self::Application(application) => w
                .build(&BlockHeader::new(*is_last, application)?)
                .and_then(|()| w.build(application).map_err(Error::Io)),
            Self::SeekTable(seektable) => w
                .build(&BlockHeader::new(*is_last, seektable)?)
                .and_then(|()| w.build(seektable)),
            Self::VorbisComment(vorbis_comment) => w
                .build(&BlockHeader::new(*is_last, vorbis_comment)?)
                .and_then(|()| w.build(vorbis_comment)),
            Self::Cuesheet(cuesheet) => w
                .build(&BlockHeader::new(*is_last, cuesheet)?)
                .and_then(|()| w.build(cuesheet)),
            Self::Picture(picture) => w
                .build(&BlockHeader::new(*is_last, picture)?)
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

macro_rules! as_block_ref {
    ($t:ty, $b:ident) => {
        impl AsBlockRef for $t {
            fn as_block_ref(&self) -> BlockRef<'_> {
                BlockRef::$b(self)
            }
        }
    };
}

as_block_ref!(Streaminfo, Streaminfo);
as_block_ref!(Padding, Padding);
as_block_ref!(Application, Application);
as_block_ref!(SeekTable, SeekTable);
as_block_ref!(VorbisComment, VorbisComment);
as_block_ref!(Cuesheet, Cuesheet);
as_block_ref!(Picture, Picture);

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

impl ToBitStreamWith<'_> for BlockRef<'_> {
    type Context = bool;
    type Error = Error;

    // builds to writer with header
    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W, is_last: &bool) -> Result<(), Error> {
        match self {
            Self::Streaminfo(streaminfo) => w
                .build(&BlockHeader::new(*is_last, *streaminfo)?)
                .and_then(|()| w.build(*streaminfo).map_err(Error::Io)),
            Self::Padding(padding) => w
                .build(&BlockHeader::new(*is_last, *padding)?)
                .and_then(|()| w.build(*padding).map_err(Error::Io)),
            Self::Application(application) => w
                .build(&BlockHeader::new(*is_last, *application)?)
                .and_then(|()| w.build(*application).map_err(Error::Io)),
            Self::SeekTable(seektable) => w
                .build(&BlockHeader::new(*is_last, *seektable)?)
                .and_then(|()| w.build(*seektable)),
            Self::VorbisComment(vorbis_comment) => w
                .build(&BlockHeader::new(*is_last, *vorbis_comment)?)
                .and_then(|()| w.build(*vorbis_comment)),
            Self::Cuesheet(cuesheet) => w
                .build(&BlockHeader::new(*is_last, *cuesheet)?)
                .and_then(|()| w.build(*cuesheet)),
            Self::Picture(picture) => w
                .build(&BlockHeader::new(*is_last, *picture)?)
                .and_then(|()| w.build(*picture)),
        }
    }
}

/// Any possible FLAC metadata block set
///
/// Certain metadata blocks may occur multiple times
/// within a FLAC metadata block list, while others
/// may only occur once.  This enforces that restriction
/// at the type level.
pub enum BlockSet {
    /// STREAMINFO may only occur once
    Streaminfo(Streaminfo),
    /// PADDING may occur multiple times
    Padding(Vec<Padding>),
    /// APPLICATION may occur multiple times
    Application(Vec<Application>),
    /// SEEKTABLE may only occur once
    SeekTable(SeekTable),
    /// VORBIS_COMMENT may only occur once
    VorbisComment(VorbisComment),
    /// CUESHEET may occur multiple times
    Cuesheet(Vec<Cuesheet>),
    /// PICTURE may occur multiple times
    Picture(Vec<Picture>),
}

impl BlockSet {
    /// Iterates over all blocks in set
    pub fn iter(&self) -> Box<dyn Iterator<Item = BlockRef<'_>> + '_> {
        use std::iter::once;

        match self {
            Self::Streaminfo(s) => Box::new(once(BlockRef::Streaminfo(s))),
            Self::Padding(p) => Box::new(p.iter().map(BlockRef::Padding)),
            Self::Application(a) => Box::new(a.iter().map(BlockRef::Application)),
            Self::SeekTable(s) => Box::new(once(BlockRef::SeekTable(s))),
            Self::VorbisComment(v) => Box::new(once(BlockRef::VorbisComment(v))),
            Self::Cuesheet(c) => Box::new(c.iter().map(BlockRef::Cuesheet)),
            Self::Picture(p) => Box::new(p.iter().map(BlockRef::Picture)),
        }
    }
}

/// A STREAMINFO metadata block
///
/// # Important
///
/// Changing any of these values to something that differs
/// from the values of the file's frame headers will render it
/// unplayable, as will moving it anywhere but the first
/// metadata block in the file.
/// Avoid modifying the position and contents of this block unless you
/// know exactly what you are doing.
#[derive(Debug, Clone)]
pub struct Streaminfo {
    /// The minimum block size (in samples) used in the stream,
    /// excluding the last block.
    pub minimum_block_size: u16,
    /// The maximum block size (in samples) used in the stream,
    /// excluding the last block.
    pub maximum_block_size: u16,
    /// The minimum framesize (in bytes) used in the stream.
    /// `None` indicates the value is unknown.
    pub minimum_frame_size: Option<NonZero<u32>>,
    /// The maximum framesize (in bytes) used in the stream.
    /// `None` indicates the value is unknown.
    pub maximum_frame_size: Option<NonZero<u32>>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: NonZero<u8>,
    /// Number of bits-per-sample, from 4 to 32
    pub bits_per_sample: SignedBitCount<32>,
    /// Total number of interchannel samples in stream.
    /// `None` indicates the value is unknown.
    pub total_samples: Option<NonZero<u64>>,
    /// MD5 hash of unencoded audio data.
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

impl MetadataBlock for Streaminfo {
    const TYPE: BlockType = BlockType::Streaminfo;
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
#[derive(Debug, Clone)]
pub struct Padding {
    /// The size of the padding, in bytes
    pub size: BlockSize,
}

impl MetadataBlock for Padding {
    const TYPE: BlockType = BlockType::Padding;
}

impl FromBitStreamWith<'_> for Padding {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: &BlockSize) -> Result<Self, Self::Error> {
        r.skip(size.get() * 8)?;
        Ok(Self { size: *size })
    }
}

impl ToBitStream for Padding {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.pad(self.size.get() * 8)
    }
}

/// An APPLICATION metadata block
#[derive(Debug, Clone)]
pub struct Application {
    /// A registered application ID
    pub id: u32,
    /// Application-specific data
    pub data: Vec<u8>,
}

impl MetadataBlock for Application {
    const TYPE: BlockType = BlockType::Application;
}

impl FromBitStreamWith<'_> for Application {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: &BlockSize) -> Result<Self, Self::Error> {
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
#[derive(Debug, Clone)]
pub struct SeekTable {
    /// The seek table's individual seek points
    pub points: Vec<SeekPoint>,
}

impl MetadataBlock for SeekTable {
    const TYPE: BlockType = BlockType::SeekTable;
}

impl FromBitStreamWith<'_> for SeekTable {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: &BlockSize) -> Result<Self, Self::Error> {
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

impl MetadataBlock for VorbisComment {
    const TYPE: BlockType = BlockType::VorbisComment;
}

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
#[derive(Debug, Clone)]
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

impl MetadataBlock for Cuesheet {
    const TYPE: BlockType = BlockType::Cuesheet;
}

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
                .map(|_| r.parse_with::<CuesheetTrack>(&is_cdda))
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
            .try_for_each(|track| w.build_with(track, &self.is_cdda))
    }
}

/// An individual CUESHEET track
#[derive(Debug, Clone)]
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

impl FromBitStreamWith<'_> for CuesheetTrack {
    type Error = Error;
    type Context = bool;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, is_cdda: &bool) -> Result<Self, Self::Error> {
        let offset: u64 = r.read_to()?;
        if *is_cdda && offset % 588 != 0 {
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

        match TrackType::new(*is_cdda, number) {
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
            let point: CuesheetIndexPoint = r.parse_with(is_cdda)?;
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

impl ToBitStreamWith<'_> for CuesheetTrack {
    type Error = Error;
    type Context = bool;

    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        is_cdda: &bool,
    ) -> Result<(), Self::Error> {
        if *is_cdda && self.offset % 588 != 0 {
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

        match TrackType::new(*is_cdda, self.number) {
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
                                    w.build_with(point, is_cdda)
                                }
                                _ => Err(Error::InvalidCuesheetIndexPointNum),
                            },
                            Some(previous_number) => match point.number == *previous_number + 1 {
                                true => {
                                    *previous_number = point.number;
                                    w.build_with(point, is_cdda)
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
#[derive(Debug, Clone)]
pub struct CuesheetIndexPoint {
    /// Offset in samples
    pub offset: u64,
    /// Track index point number
    pub number: u8,
}

impl FromBitStreamWith<'_> for CuesheetIndexPoint {
    type Error = Error;
    type Context = bool;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, is_cdda: &bool) -> Result<Self, Self::Error> {
        let offset: u64 = r.read_to()?;
        if *is_cdda && offset % 588 != 0 {
            return Err(Error::InvalidCuesheetOffset);
        }
        let number = r.read_to()?;
        r.skip(3 * 8)?;
        Ok(Self { offset, number })
    }
}

impl ToBitStreamWith<'_> for CuesheetIndexPoint {
    type Error = Error;
    type Context = bool;

    fn to_writer<W: BitWrite + ?Sized>(
        &self,
        w: &mut W,
        is_cdda: &bool,
    ) -> Result<(), Self::Error> {
        match *is_cdda && self.offset % 588 != 0 {
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
    pub colors_used: u32,
    /// The binary picture data
    pub data: Vec<u8>,
}

impl MetadataBlock for Picture {
    const TYPE: BlockType = BlockType::Picture;
}

impl Picture {
    /// Attempt to create a new PICTURE block from raw image data
    ///
    /// # Errors
    ///
    /// Returns an error if some problem occurs reading
    /// or identifying the file.
    pub fn new(
        picture_type: PictureType,
        description: String,
        data: Vec<u8>,
    ) -> Result<Self, InvalidPicture> {
        let metrics = PictureMetrics::try_new(&data)?;
        Ok(Self {
            picture_type,
            description,
            data,
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
            colors_used: r.read_to()?,
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
        w.write_from(self.colors_used)?;
        prefixed_field(w, &self.data, Error::ExcessivePictureSize)
    }
}

/// Defined variants of PICTURE type
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PictureType {
    /// Other
    Other,
    /// PNG file icon of 32x32 pixels
    Png32x32,
    /// General file icon
    GeneralFileIcon,
    /// Front cover
    FrontCover,
    /// Back cover
    BackCover,
    /// Liner notes page
    LinerNotes,
    /// Media label (e.g., CD, Vinyl or Cassette label)
    MediaLabel,
    /// Lead artist, lead performer, or soloist
    LeadArtist,
    /// Artist or performer
    Artist,
    /// Conductor
    Conductor,
    /// Band or orchestra
    Band,
    /// Composer
    Composer,
    /// Lyricist or text writer
    Lyricist,
    /// Recording location
    RecordingLocation,
    /// During recording
    DuringRecording,
    /// During performance
    DuringPerformance,
    /// Movie or video screen capture
    ScreenCapture,
    /// A bright colored fish
    Fish,
    /// Illustration
    Illustration,
    /// Band or artist logotype
    BandLogo,
    /// Publisher or studio logotype
    PublisherLogo,
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
    colors_used: u32,
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
            0 => (bit_depth.into(), 0),       // grayscale
            2 => ((bit_depth * 3).into(), 0), // RGB
            3 => (0, plte_colors(r)?),        // palette
            4 => ((bit_depth * 2).into(), 0), // grayscale + alpha
            6 => ((bit_depth * 4).into(), 0), // RGB + alpha
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
                        colors_used: 0,
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
            colors_used: 1 << (r.read::<3, u32>()? + 1),
            color_depth: 0,
        })
    }
}
