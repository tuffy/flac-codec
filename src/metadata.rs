use crate::Error;
use bitstream_io::{
    BigEndian, BitRead, BitReader, BitWrite, FromBitStream, FromBitStreamWith, LittleEndian,
    ToBitStream,
};
use std::num::NonZero;

#[derive(Debug)]
pub struct BlockHeader {
    last: bool,
    block_type: BlockType,
    size: BlockSize,
}

impl FromBitStream for BlockHeader {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        Ok(Self {
            last: r.read_bit()?,
            block_type: r.parse()?,
            size: r.parse()?,
        })
    }
}

impl ToBitStream for BlockHeader {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_bit(self.last)?;
        w.build(&self.block_type)?;
        w.build(&self.size)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum BlockType {
    Streaminfo,
    Padding,
    Application,
    SeekTable,
    VorbisComment,
    Cuesheet,
    Picture,
}

impl FromBitStream for BlockType {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        match r.read_in::<7, u8>()? {
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
        w.write_out::<7, u8>(match self {
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

/// a 24-bit block size value, with safeguards against overflow
#[derive(Debug, Default, Copy, Clone)]
pub struct BlockSize(u32);

impl BlockSize {
    const MAX: u32 = (1 << 24) - 1;

    fn get(&self) -> u32 {
        self.0
    }

    // fn checked_add(self, Self(rhs): Self) -> Result<Self, BlockSizeOverflow> {
    //     self.0
    //         .checked_add(rhs)
    //         .filter(|s| *s <= Self::MAX)
    //         .map(Self)
    //         .ok_or(BlockSizeOverflow)
    // }
}

impl FromBitStream for BlockSize {
    type Error = std::io::Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        r.read_in::<24, _>().map(Self)
    }
}

impl ToBitStream for BlockSize {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_out::<24, _>(self.0)
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

pub struct BlockSizeOverflow;

pub struct BlockReader<R: std::io::Read> {
    reader: R,
    failed: bool,
    tag_read: bool,
    streaminfo_read: bool,
    seektable_read: bool,
    png_read: bool,
    icon_read: bool,
    finished: bool,
}

impl<R: std::io::Read> BlockReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            failed: false,
            tag_read: false,
            streaminfo_read: false,
            seektable_read: false,
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
                    if self.seektable_read {
                        self.failed = true;
                        Some(Err(Error::MultipleSeekTable))
                    } else {
                        self.seektable_read = true;
                        seektable
                    }
                }
                picture @ Some(Ok(Block::Picture(Picture {
                    picture_type: PictureType::Png32x32,
                    ..
                }))) => {
                    if self.png_read {
                        self.failed = true;
                        Some(Err(Error::MultiplePngIcon))
                    } else {
                        self.png_read = true;
                        picture
                    }
                }
                picture @ Some(Ok(Block::Picture(Picture {
                    picture_type: PictureType::GeneralFileIcon,
                    ..
                }))) => {
                    if self.icon_read {
                        self.failed = true;
                        Some(Err(Error::MultipleGeneralIcon))
                    } else {
                        self.icon_read = true;
                        picture
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

#[derive(Debug, Clone)]
pub enum Block {
    Streaminfo(Streaminfo),
    Padding(Padding),
    Application(Application),
    SeekTable(SeekTable),
    VorbisComment(VorbisComment),
    Cuesheet(Cuesheet),
    Picture(Picture),
}

impl FromBitStreamWith<'_> for Block {
    type Context = BlockHeader;
    type Error = Error;

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

#[derive(Debug, Clone)]
pub struct Streaminfo {
    pub minimum_block_size: u16,
    pub maximum_block_size: u16,
    pub minimum_frame_size: Option<NonZero<u32>>,
    pub maximum_frame_size: Option<NonZero<u32>>,
    pub sample_rate: u32,
    pub channels: NonZero<u8>,
    pub bits_per_sample: NonZero<u8>,
    pub total_samples: Option<NonZero<u64>>,
    pub md5: Option<[u8; 16]>,
}

impl FromBitStream for Streaminfo {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        Ok(Self {
            minimum_block_size: r.read_to()?,
            maximum_block_size: r.read_to()?,
            minimum_frame_size: NonZero::new(r.read_in::<24, _>()?),
            maximum_frame_size: NonZero::new(r.read_in::<24, _>()?),
            sample_rate: r.read_in::<20, _>()?,
            channels: NonZero::new(r.read_in::<3, u8>()? + 1).unwrap(),
            bits_per_sample: NonZero::new(r.read_in::<5, u8>()? + 1).unwrap(),
            total_samples: NonZero::new(r.read_in::<36, u64>()?),
            md5: r
                .read_to()
                .map(|md5: [u8; 16]| md5.iter().any(|b| *b != 0).then_some(md5))?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Padding {
    pub size: BlockSize,
}

impl FromBitStreamWith<'_> for Padding {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: &BlockSize) -> Result<Self, Self::Error> {
        r.skip(size.get() * 8)?;
        Ok(Self { size: *size })
    }
}

#[derive(Debug, Clone)]
pub struct Application {
    pub id: u32,
    pub data: Vec<u8>,
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

#[derive(Debug, Clone)]
pub struct SeekTable {
    pub points: Vec<SeekPoint>,
}

impl FromBitStreamWith<'_> for SeekTable {
    type Context = BlockSize;
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R, size: &BlockSize) -> Result<Self, Self::Error> {
        match (size.get().checked_div(18), size.get() % 18) {
            (Some(p), 0) => {
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

#[derive(Debug, Clone)]
pub struct SeekPoint {
    pub sample_offset: Option<u64>,
    pub byte_offset: u64,
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

#[derive(Debug, Clone)]
pub struct VorbisComment {
    pub vendor_string: String,
    pub fields: Vec<String>,
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

#[derive(Debug, Clone)]
pub struct Cuesheet {
    pub catalog_number: Box<[u8; 128]>,
    pub lead_in_samples: u64,
    pub is_cdda: bool,
    pub tracks: Vec<CuesheetTrack>,
}

impl FromBitStream for Cuesheet {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let catalog_number = r.read_to()?;
        let lead_in_samples = r.read_to()?;
        let is_cdda = r.read_bit()?;
        r.skip(7 + 258 * 8)?;
        let track_count = r.read_to::<u8>()?;
        let tracks = (0..track_count)
            .map(|_| r.parse_with::<CuesheetTrack>(&is_cdda))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            catalog_number: Box::new(catalog_number),
            lead_in_samples,
            is_cdda,
            tracks,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CuesheetTrack {
    pub offset: u64,
    pub number: u8,
    pub isrc: Option<[u8; 12]>,
    pub non_audio: bool,
    pub pre_emphasis: bool,
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
        let non_audio = r.read_bit()?;
        let pre_emphasis = r.read_bit()?;
        r.skip(6 + 13 * 8)?;
        let point_count = r.read_to::<u8>()?;

        match *is_cdda {
            true => match number {
                1..=99 => {
                    if point_count == 0 {
                        return Err(Error::InvalidCuesheetIndexPoints);
                    }
                }
                170 => {
                    if point_count != 0 {
                        return Err(Error::InvalidCuesheetIndexPoints);
                    }
                }
                _ => {
                    return Err(Error::InvalidCuesheetTrackNumber);
                }
            },
            false => {
                if (number == 255) && (point_count != 0) {
                    return Err(Error::InvalidCuesheetIndexPoints);
                }
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

#[derive(Debug, Clone)]
pub struct CuesheetIndexPoint {
    pub offset: u64,
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

#[derive(Debug, Clone)]
pub struct Picture {
    pub picture_type: PictureType,
    pub media_type: String,
    pub description: String,
    pub width: u32,
    pub height: u32,
    pub color_depth: u32,
    pub colors_used: u32,
    pub data: Vec<u8>,
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

#[derive(Debug, Clone)]
pub enum PictureType {
    Other,
    Png32x32,
    GeneralFileIcon,
    FrontCover,
    BackCover,
    LinerNotes,
    MediaLabel,
    LeadArtist,
    Artist,
    Conductor,
    Band,
    Composer,
    Lyricist,
    RecordingLocation,
    DuringRecording,
    DuringPerformance,
    ScreenCapture,
    Fish,
    Illustration,
    BandLogo,
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
