use crate::Error;
use crate::metadata::CuesheetError;
use crate::metadata::contiguous::{Adjacent, Contiguous};
use bitstream_io::{BitRead, BitWrite, FromBitStream, ToBitStream};
use std::num::NonZero;
use std::str::FromStr;

/// An ASCII digit, for the catalog number
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Digit {
    /// U+0030
    Digit0 = 48,
    /// U+0031
    Digit1 = 49,
    /// U+0032
    Digit2 = 50,
    /// U+0033
    Digit3 = 51,
    /// U+0034
    Digit4 = 52,
    /// U+0035
    Digit5 = 53,
    /// U+0036
    Digit6 = 54,
    /// U+0037
    Digit7 = 55,
    /// U+0038
    Digit8 = 56,
    /// U+0039
    Digit9 = 57,
}

impl TryFrom<u8> for Digit {
    type Error = u8;

    fn try_from(u: u8) -> Result<Digit, u8> {
        match u {
            48 => Ok(Self::Digit0),
            49 => Ok(Self::Digit1),
            50 => Ok(Self::Digit2),
            51 => Ok(Self::Digit3),
            52 => Ok(Self::Digit4),
            53 => Ok(Self::Digit5),
            54 => Ok(Self::Digit6),
            55 => Ok(Self::Digit7),
            56 => Ok(Self::Digit8),
            57 => Ok(Self::Digit9),
            u => Err(u),
        }
    }
}

impl TryFrom<char> for Digit {
    type Error = CuesheetError;

    fn try_from(c: char) -> Result<Digit, CuesheetError> {
        match c {
            '0' => Ok(Self::Digit0),
            '1' => Ok(Self::Digit1),
            '2' => Ok(Self::Digit2),
            '3' => Ok(Self::Digit3),
            '4' => Ok(Self::Digit4),
            '5' => Ok(Self::Digit5),
            '6' => Ok(Self::Digit6),
            '7' => Ok(Self::Digit7),
            '8' => Ok(Self::Digit8),
            '9' => Ok(Self::Digit9),
            _ => Err(CuesheetError::InvalidCatalogNumber),
        }
    }
}

impl From<Digit> for u8 {
    fn from(d: Digit) -> u8 {
        d as u8
    }
}

impl std::fmt::Display for Digit {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Digit0 => '0'.fmt(f),
            Self::Digit1 => '1'.fmt(f),
            Self::Digit2 => '2'.fmt(f),
            Self::Digit3 => '3'.fmt(f),
            Self::Digit4 => '4'.fmt(f),
            Self::Digit5 => '5'.fmt(f),
            Self::Digit6 => '6'.fmt(f),
            Self::Digit7 => '7'.fmt(f),
            Self::Digit8 => '8'.fmt(f),
            Self::Digit9 => '9'.fmt(f),
        }
    }
}

/// An offset for CD-DA
///
/// These must be evenly divisible by 588 samples
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct CDDAOffset {
    offset: u64,
}

impl CDDAOffset {
    const SAMPLES_PER_SECTOR: u64 = 44100 / 75;
}

impl std::ops::Sub for CDDAOffset {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            offset: self.offset - rhs.offset,
        }
    }
}

impl FromStr for CDDAOffset {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        let (mm, rest) = s.split_once(':').ok_or(())?;
        let (ss, ff) = rest.split_once(':').ok_or(())?;

        let ff: u64 = ff.parse().ok().filter(|ff| *ff < 75).ok_or(())?;
        let ss: u64 = ss.parse().ok().filter(|ss| *ss < 60).ok_or(())?;
        let mm: u64 = mm.parse().map_err(|_| ())?;

        Ok(Self {
            offset: (ff + ss * 75 + mm * 75 * 60) * 588,
        })
    }
}

impl std::fmt::Display for CDDAOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.offset.fmt(f)
    }
}

impl From<CDDAOffset> for u64 {
    fn from(o: CDDAOffset) -> Self {
        o.offset
    }
}

impl TryFrom<u64> for CDDAOffset {
    type Error = u64;

    fn try_from(offset: u64) -> Result<Self, Self::Error> {
        ((offset % Self::SAMPLES_PER_SECTOR) == 0)
            .then_some(Self { offset })
            .ok_or(offset)
    }
}

impl std::ops::Add for CDDAOffset {
    type Output = Self;

    fn add(self, rhs: CDDAOffset) -> Self {
        // if both are already divisible by 588,
        // their added quantities will also
        // be divsible by 588
        Self {
            offset: self.offset + rhs.offset,
        }
    }
}

impl FromBitStream for CDDAOffset {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        Ok(Self {
            offset: r.read_to().map_err(Error::Io).and_then(|o| {
                ((o % Self::SAMPLES_PER_SECTOR) == 0)
                    .then_some(o)
                    .ok_or(CuesheetError::InvalidCDDAOffset.into())
            })?,
        })
    }
}

impl ToBitStream for CDDAOffset {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        // value already checked for divisibility,
        // so no need to check it again
        w.write_from(self.offset)
    }
}

impl Adjacent for CDDAOffset {
    fn valid_first(&self) -> bool {
        self.offset == 0
    }

    fn is_next(&self, previous: &Self) -> bool {
        self.offset > previous.offset
    }
}

/// The track number for lead-out tracks
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LeadOut;

impl LeadOut {
    /// Lead-out track number for CD-DA discs
    pub const CDDA: NonZero<u8> = NonZero::new(170).unwrap();

    /// Lead-out track number for non-CD-DA discs
    pub const NON_CDDA: NonZero<u8> = NonZero::new(255).unwrap();
}

/// An International Standard Recording Code value
///
/// These are used to assign a unique identifier
/// to sound and music video recordings.
///
/// This is a 12 character code which may be
/// delimited by optional dashes.
///
/// ```text
///  letters     digits
///       ↓↓     ↓↓
///       AA-6Q7-20-00047
///          ↑↑↑    ↑↑↑↑↑
/// alphanumeric    digits
/// ```
///
/// The first five characters are the prefix code.
/// The following two digits are the year of reference.
/// The final five digits are the designation code.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ISRCString(String);

impl std::fmt::Display for ISRCString {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl AsRef<str> for ISRCString {
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}

impl FromStr for ISRCString {
    type Err = CuesheetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use std::borrow::Cow;

        fn filter_split(s: &str, amt: usize, f: impl Fn(char) -> bool) -> Option<&str> {
            s.split_at_checked(amt)
                .and_then(|(prefix, rest)| prefix.chars().all(f).then_some(rest))
        }

        // strip out dashes if necessary
        let isrc: Cow<'_, str> = if s.contains('-') {
            s.chars().filter(|c| *c != '-').collect::<String>().into()
        } else {
            s.into()
        };

        filter_split(&isrc, 2, |c| c.is_ascii_alphabetic())
            .and_then(|s| filter_split(s, 3, |c| c.is_ascii_alphanumeric()))
            .and_then(|s| filter_split(s, 2, |c| c.is_ascii_digit()))
            .and_then(|s| s.chars().all(|c| c.is_ascii_digit()).then_some(()))
            .map(|()| ISRCString(isrc.into_owned()))
            .ok_or(CuesheetError::InvalidISRC)
    }
}

/// An optional ISRC value
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub enum ISRC {
    /// An undefined ISRC value in which all bits are 0
    #[default]
    None,
    /// A defined ISRC value matching the ISRC format
    String(ISRCString),
}

impl std::fmt::Display for ISRC {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::String(s) => s.fmt(f),
            Self::None => "".fmt(f),
        }
    }
}

impl FromBitStream for ISRC {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Error> {
        let isrc = r.read_to::<[u8; 12]>()?;
        if isrc.iter().all(|b| *b == 0) {
            Ok(ISRC::None)
        } else {
            let s = str::from_utf8(&isrc).map_err(|_| CuesheetError::InvalidISRC)?;

            Ok(ISRC::String(s.parse()?))
        }
    }
}

impl ToBitStream for ISRC {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), std::io::Error> {
        w.write_from(match self {
            Self::String(isrc) => {
                let mut o = [0; 12];
                o.iter_mut()
                    .zip(isrc.as_ref().as_bytes())
                    .for_each(|(o, i)| *o = *i);
                o
            }
            Self::None => [0; 12],
        })
    }
}

impl AsRef<str> for ISRC {
    fn as_ref(&self) -> &str {
        match self {
            Self::String(s) => s.as_ref(),
            Self::None => "",
        }
    }
}

impl FromStr for ISRC {
    type Err = CuesheetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ISRCString::from_str(s).map(ISRC::String)
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
pub struct Track<O, N, P> {
    /// Offset of first index point
    ///
    /// In samples relative to the beginning of the FLAC audio stream.
    ///
    /// For CD-DA, the track offset must always be divisible by 588.
    /// This is because for audio CDs, tracks must always begin
    /// on CD frame boundaries.  Since each CD frame
    /// is 1/75th of a second, and CDs have 44,100 samples per second,
    /// 44100 ÷ 75 = 588.
    ///
    /// Non-CD-DA discs have no such restriction.
    pub offset: O,

    /// Track number
    ///
    /// | Disc Type  | Range                  | Lead-Out Track
    /// |-----------:|:----------------------:|---------------
    /// | CD-DA      | 1 ≤ track number ≤ 99  | 170
    /// | Non-CD-DA  | 1 ≤ track number < 255 | 255
    pub number: N,

    /// Track's ISRC
    pub isrc: ISRC,

    /// Whether track is non-audio
    pub non_audio: bool,

    /// Whether track has pre-emphasis
    pub pre_emphasis: bool,

    /// Track's index points
    ///
    /// | Disc Type | Lead-Out Track | Index Points          |
    /// |----------:|:--------------:|-----------------------|
    /// | CD-DA     | No             | not more than 100     |
    /// | CD-DA     | Yes            | 0                     |
    /// | Non-CD-DA | No             | not more than 255     |
    /// | Non-CD-DA | Yes            | 0                     |
    pub index_points: P,
}

impl<const MAX: usize, O: Adjacent, N: Adjacent> Adjacent for Track<O, N, IndexVec<MAX, O>> {
    fn valid_first(&self) -> bool {
        self.offset.valid_first() && self.number.valid_first()
    }

    fn is_next(&self, previous: &Self) -> bool {
        self.number.is_next(&previous.number) && self.offset.is_next(previous.index_points.last())
    }
}

/// A Generic track suitable for display
///
/// The lead-out track has a track number of `None`.
pub type TrackGeneric = Track<u64, Option<u8>, Vec<Index<u64>>>;

/// A CD-DA CUESHEET track
pub type TrackCDDA = Track<CDDAOffset, NonZero<u8>, IndexVec<100, CDDAOffset>>;

impl FromBitStream for TrackCDDA {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let offset = r.parse()?;
        let number = r
            .read_to()
            .map_err(Error::Io)
            .and_then(|s| NonZero::new(s).ok_or(Error::from(CuesheetError::InvalidIndexPoint)))?;
        let isrc = r.parse()?;
        let non_audio = r.read_bit()?;
        let pre_emphasis = r.read_bit()?;
        r.skip(6 + 13 * 8)?;
        let index_point_count = r.read_to::<u8>()?;

        Ok(Self {
            offset,
            number,
            isrc,
            non_audio,
            pre_emphasis,
            // IndexVec guarantees at least 1 index point
            // Contiguous guarantees there's no more than MAX index points
            // and that they're all in order
            index_points: IndexVec::try_from(
                Contiguous::try_collect((0..index_point_count).map(|_| r.parse()))
                    .map_err(|_| Error::from(CuesheetError::IndexPointsOutOfSequence))??,
            )?,
        })
    }
}

impl ToBitStream for TrackCDDA {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.build(&self.offset)?;
        w.write_from(self.number.get())?;
        w.build(&self.isrc)?;
        w.write_bit(self.non_audio)?;
        w.write_bit(self.pre_emphasis)?;
        w.pad(6 + 13 * 8)?;
        w.write_from::<u8>(self.index_points.len().try_into().unwrap())?;
        for point in self.index_points.iter() {
            w.build(point)?;
        }
        Ok(())
    }
}

/// A non-CD-DA CUESHEET track
pub type TrackNonCDDA = Track<u64, NonZero<u8>, IndexVec<256, u64>>;

impl FromBitStream for TrackNonCDDA {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let offset = r.read_to()?;
        let number = r
            .read_to()
            .map_err(Error::Io)
            .and_then(|s| NonZero::new(s).ok_or(Error::from(CuesheetError::InvalidIndexPoint)))?;
        let isrc = r.parse()?;
        let non_audio = r.read_bit()?;
        let pre_emphasis = r.read_bit()?;
        r.skip(6 + 13 * 8)?;
        let index_point_count = r.read_to::<u8>()?;

        Ok(Self {
            offset,
            number,
            isrc,
            non_audio,
            pre_emphasis,
            // IndexVec guarantees at least 1 index point
            // Contiguous guarantees there's no more than MAX index points
            // and that they're all in order
            index_points: IndexVec::try_from(
                Contiguous::try_collect((0..index_point_count).map(|_| r.parse()))
                    .map_err(|_| Error::from(CuesheetError::IndexPointsOutOfSequence))??,
            )?,
        })
    }
}

impl ToBitStream for TrackNonCDDA {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(self.offset)?;
        w.write_from(self.number.get())?;
        w.build(&self.isrc)?;
        w.write_bit(self.non_audio)?;
        w.write_bit(self.pre_emphasis)?;
        w.pad(6 + 13 * 8)?;
        w.write_from::<u8>(self.index_points.len().try_into().unwrap())?;
        for point in self.index_points.iter() {
            w.build(point)?;
        }
        Ok(())
    }
}

/// A CD-DA CUESHEET lead-out track
pub type LeadOutCDDA = Track<CDDAOffset, LeadOut, ()>;

impl FromBitStream for LeadOutCDDA {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let offset = r.parse()?;
        let number = r.read_to::<u8>().map_err(Error::Io).and_then(|n| {
            NonZero::new(n)
                .filter(|n| *n == LeadOut::CDDA)
                .map(|_| LeadOut)
                .ok_or(CuesheetError::TracksOutOfSequence.into())
        })?;
        let isrc = r.parse()?;
        let non_audio = r.read_bit()?;
        let pre_emphasis = r.read_bit()?;
        r.skip(6 + 13 * 8)?;
        match r.read_to::<u8>()? {
            0 => Ok(Self {
                offset,
                number,
                isrc,
                non_audio,
                pre_emphasis,
                index_points: (),
            }),
            // because parsing a cuesheet generates a lead-out
            // automatically, this error can only only occur when
            // reading from metadata blocks
            _ => Err(CuesheetError::IndexPointsInLeadout.into()),
        }
    }
}

impl ToBitStream for LeadOutCDDA {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.build(&self.offset)?;
        w.write_from(LeadOut::CDDA.get())?;
        w.build(&self.isrc)?;
        w.write_bit(self.non_audio)?;
        w.write_bit(self.pre_emphasis)?;
        w.pad(6 + 13 * 8)?;
        w.write_from::<u8>(0)?;
        Ok(())
    }
}

impl LeadOutCDDA {
    /// Creates new lead-out track with the given offset
    ///
    /// Lead-out offset must be contiguous with existing tracks
    pub fn new(last: Option<&TrackCDDA>, offset: CDDAOffset) -> Result<Self, CuesheetError> {
        match last {
            Some(track) if *track.index_points.last() >= offset => Err(CuesheetError::ShortLeadOut),
            _ => Ok(LeadOutCDDA {
                offset,
                number: LeadOut,
                isrc: ISRC::None,
                non_audio: false,
                pre_emphasis: false,
                index_points: (),
            }),
        }
    }
}

/// A non-CD-DA CUESHEET lead-out track
pub type LeadOutNonCDDA = Track<u64, LeadOut, ()>;

impl FromBitStream for LeadOutNonCDDA {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let offset = r.read_to()?;
        let number = r.read_to::<u8>().map_err(Error::Io).and_then(|n| {
            NonZero::new(n)
                .filter(|n| *n == LeadOut::NON_CDDA)
                .map(|_| LeadOut)
                .ok_or(CuesheetError::TracksOutOfSequence.into())
        })?;
        let isrc = r.parse()?;
        let non_audio = r.read_bit()?;
        let pre_emphasis = r.read_bit()?;
        r.skip(6 + 13 * 8)?;
        match r.read_to::<u8>()? {
            0 => Ok(Self {
                offset,
                number,
                isrc,
                non_audio,
                pre_emphasis,
                index_points: (),
            }),
            // because parsing a cuesheet generates a lead-out
            // automatically, this error can only only occur when
            // reading from metadata blocks
            _ => Err(CuesheetError::IndexPointsInLeadout.into()),
        }
    }
}

impl ToBitStream for LeadOutNonCDDA {
    type Error = Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(self.offset)?;
        w.write_from::<u8>(LeadOut::NON_CDDA.get())?;
        w.build(&self.isrc)?;
        w.write_bit(self.non_audio)?;
        w.write_bit(self.pre_emphasis)?;
        w.pad(6 + 13 * 8)?;
        w.write_from::<u8>(0)?;
        Ok(())
    }
}

impl LeadOutNonCDDA {
    /// Creates new lead-out track with the given offset
    pub fn new(last: Option<&TrackNonCDDA>, offset: u64) -> Result<Self, CuesheetError> {
        match last {
            Some(track) if *track.index_points.last() >= offset => Err(CuesheetError::ShortLeadOut),
            _ => Ok(LeadOutNonCDDA {
                offset,
                number: LeadOut,
                isrc: ISRC::None,
                non_audio: false,
                pre_emphasis: false,
                index_points: (),
            }),
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
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Index<O> {
    /// Offset in samples from beginning of track
    pub offset: O,

    /// Track index point number
    pub number: u8,
}

impl<O: Adjacent> Adjacent for Index<O> {
    fn valid_first(&self) -> bool {
        self.offset.valid_first() && matches!(self.number, 0 | 1)
    }

    fn is_next(&self, previous: &Self) -> bool {
        self.offset.is_next(&previous.offset) && self.number == previous.number + 1
    }
}

impl FromBitStream for Index<CDDAOffset> {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let offset = r.parse()?;
        let number = r.read_to()?;
        r.skip(3 * 8)?;
        Ok(Self { offset, number })
    }
}

impl FromBitStream for Index<u64> {
    type Error = Error;

    fn from_reader<R: BitRead + ?Sized>(r: &mut R) -> Result<Self, Self::Error> {
        let offset = r.read_to()?;
        let number = r.read_to()?;
        r.skip(3 * 8)?;
        Ok(Self { offset, number })
    }
}

impl ToBitStream for Index<CDDAOffset> {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.build(&self.offset)?;
        w.write_from(self.number)?;
        w.pad(3 * 8)
    }
}

impl ToBitStream for Index<u64> {
    type Error = std::io::Error;

    fn to_writer<W: BitWrite + ?Sized>(&self, w: &mut W) -> Result<(), Self::Error> {
        w.write_from(self.offset)?;
        w.write_from(self.number)?;
        w.pad(3 * 8)
    }
}

/// A Vec of Indexes with the given offset type
///
/// Tracks other than the lead-out are required
/// to have at least one `INDEX 01` index point,
/// which specifies the beginning of the track.
/// An `INDEX 00` pre-gap point is optional.
///
/// `MAX` is the maximum number of index points
/// this can hold, including the first.
/// This is 100 for CD-DA (`00` to `99`, inclusive)
/// and 254 for non-CD-DA cuesheets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexVec<const MAX: usize, O: Adjacent> {
    // pre-gap
    index_00: Option<Index<O>>,
    // start of track
    index_01: Index<O>,
    // remaining index points
    remainder: Box<[Index<O>]>,
}

impl<const MAX: usize, O: Adjacent> IndexVec<MAX, O> {
    /// Returns number of `Index` points in `IndexVec`
    // This method never returns 0, so cannot be empty,
    // so it doesn't make sense to implement is_empty()
    // for it because it would always return false.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        // because we're created from a Contiguous Vec
        // whose size must be <= usize,
        // our len is 1 less than usize, so len() + 1
        // can never overflow
        usize::from(self.index_00.is_some()) + 1 + self.remainder.len()
    }

    /// Iterates over shared references of all `Index` points
    pub fn iter(&self) -> impl Iterator<Item = &Index<O>> {
        self.index_00
            .iter()
            .chain(std::iter::once(&self.index_01))
            .chain(&self.remainder)
    }

    /// Returns offset of track pre-gap, any
    ///
    /// This corresponds to `INDEX 00`
    pub fn pre_gap(&self) -> Option<&O> {
        match &self.index_00 {
            Some(Index { offset, .. }) => Some(offset),
            None => None,
        }
    }

    /// Returns offset of track start
    ///
    /// This corresponds to `INDEX 01`
    pub fn start(&self) -> &O {
        &self.index_01.offset
    }

    /// Returns shared reference to final item
    ///
    /// Since `IndexVec` must always contain at least
    /// one item, this method is infallible
    pub fn last(&self) -> &O {
        match self.remainder.last() {
            Some(Index { offset, .. }) => offset,
            None => self.start(),
        }
    }
}

impl<const MAX: usize, O: Adjacent> TryFrom<Contiguous<MAX, Index<O>>> for IndexVec<MAX, O> {
    type Error = CuesheetError;

    fn try_from(items: Contiguous<MAX, Index<O>>) -> Result<Self, CuesheetError> {
        use std::collections::VecDeque;

        let mut items: VecDeque<Index<O>> = items.into();

        match items.pop_front().ok_or(CuesheetError::NoIndexPoints)? {
            index_00 @ Index { number: 0, .. } => Ok(Self {
                index_00: Some(index_00),
                index_01: items
                    .pop_front()
                    .filter(|i| i.number == 1)
                    .ok_or(CuesheetError::IndexPointsOutOfSequence)?,
                remainder: Vec::from(items).into_boxed_slice(),
            }),
            index_01 @ Index { number: 1, .. } => Ok(Self {
                index_00: None,
                index_01,
                remainder: Vec::from(items).into_boxed_slice(),
            }),
            Index { .. } => Err(CuesheetError::IndexPointsOutOfSequence),
        }
    }
}
