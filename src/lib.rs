pub mod metadata;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    MissingFlacTag,
    MissingStreaminfo,
    MultipleStreaminfo,
    MultipleSeekTable,
    InvalidSeekTableSize,
    InvalidSeekTablePoint,
    InvalidCuesheetOffset,
    InvalidCuesheetTrackNumber,
    InvalidCuesheetIndexPoints,
    InvalidCuesheetIndexPointNum,
    InvalidPictureType,
    MultiplePngIcon,
    MultipleGeneralIcon,
    ReservedMetadataBlock,
    InvalidMetadataBlock,
    InvalidMetadataBlockSize,
    InsufficientApplicationBlock,
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::Utf8(error)
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Utf8(e) => e.fmt(f),
            Self::MissingFlacTag => "missing FLAC tag".fmt(f),
            Self::MissingStreaminfo => "STREAMINFO block not first in file".fmt(f),
            Self::MultipleStreaminfo => "multiple STREAMINFO blocks found in file".fmt(f),
            Self::MultipleSeekTable => "multiple SEEKTABLE blocks found in file".fmt(f),
            Self::InvalidSeekTableSize => "invalid SEEKTABLE block size".fmt(f),
            Self::InvalidSeekTablePoint => "invalid SEEKTABLE point".fmt(f),
            Self::InvalidCuesheetOffset => "invalid CUESHEET sample offset".fmt(f),
            Self::InvalidCuesheetTrackNumber => "invalid CUESHEET track number".fmt(f),
            Self::InvalidCuesheetIndexPoints => {
                "invalid number of CUESHEET track index points".fmt(f)
            }
            Self::InvalidCuesheetIndexPointNum => "invalid CUESHEET index point number".fmt(f),
            Self::InvalidPictureType => "reserved PICTURE type".fmt(f),
            Self::MultiplePngIcon => "multiple PNG icons in PICTURE blocks".fmt(f),
            Self::MultipleGeneralIcon => "multiple general file icons in PICTURE blocks".fmt(f),
            Self::ReservedMetadataBlock => "reserved metadata block".fmt(f),
            Self::InvalidMetadataBlock => "invalid metadata block".fmt(f),
            Self::InvalidMetadataBlockSize => "invalid metadat block size".fmt(f),
            Self::InsufficientApplicationBlock => "APPLICATION block too small for data".fmt(f),
        }
    }
}
