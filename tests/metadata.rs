use flac_codec::Error;
use flac_codec::metadata::{Block, VorbisComment, read_blocks, write_blocks};

fn roundtrip_test(flac: &[u8]) {
    use std::io::Read;

    let mut data = std::io::Cursor::new(flac);
    let blocks = dbg!(
        read_blocks(data.by_ref())
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    );
    let mut output = Vec::new();
    write_blocks(&mut output, blocks).unwrap();
    std::io::copy(&mut data, &mut output).unwrap();
    assert_eq!(flac.len(), output.len());
    assert_eq!(flac, output.as_slice());
}

#[test]
fn test_block_roundtrips() {
    roundtrip_test(include_bytes!("data/all-frames.flac").as_slice());
    roundtrip_test(include_bytes!("data/seektable.flac").as_slice());
    roundtrip_test(include_bytes!("data/cuesheet.flac").as_slice());
    roundtrip_test(include_bytes!("data/comment.flac").as_slice());
    roundtrip_test(include_bytes!("data/picture.flac").as_slice());
}

fn perform_test(flac: &[u8], f: impl FnOnce(&mut Vec<Block>)) -> Result<(), Error> {
    let mut blocks = read_blocks(flac).collect::<Result<Vec<_>, _>>().unwrap();

    f(&mut blocks);

    write_blocks(std::io::sink(), blocks)?;

    Ok(())
}

fn basic_test(f: impl FnOnce(&mut Vec<Block>)) -> Result<(), Error> {
    perform_test(include_bytes!("data/all-frames.flac").as_slice(), f)
}

fn seektable_test(f: impl FnOnce(&mut Vec<Block>)) -> Result<(), Error> {
    perform_test(include_bytes!("data/seektable.flac").as_slice(), f)
}

#[test]
fn test_write_metadata() {
    use flac_codec::metadata::fields::TITLE;

    assert!(matches!(basic_test(|_| { /* do nothing */ }), Ok(())));

    // STREAMINFO must be present
    assert!(matches!(
        basic_test(|blocks| {
            blocks.pop();
        }),
        Err(Error::MissingStreaminfo)
    ));

    // only one STREAMINFO allowed
    assert!(matches!(
        basic_test(|blocks| {
            let streaminfo = blocks[0].clone();
            blocks.push(streaminfo);
        }),
        Err(Error::MultipleStreaminfo)
    ));

    assert!(matches!(
        basic_test(|blocks| {
            let mut comment = VorbisComment::default();
            comment.insert(TITLE, "Test Title");
            blocks.push(comment.into());
        }),
        Ok(())
    ));

    // STREAMINFO must always be first
    assert!(matches!(
        basic_test(|blocks| {
            let mut comment = VorbisComment::default();
            comment.insert(TITLE, "Test Title");
            blocks.insert(0, comment.into());
        }),
        Err(Error::MissingStreaminfo)
    ));

    // only one VORBIS_COMMENT allowed
    assert!(matches!(
        basic_test(|blocks| {
            let mut comment = VorbisComment::default();
            comment.insert(TITLE, "Test Title");
            blocks.push(comment.clone().into());
            blocks.push(comment.into());
        }),
        Err(Error::MultipleVorbisComment)
    ));

    assert!(matches!(seektable_test(|_| { /* do nothing */ }), Ok(())));

    // only one SEEKTABLE allowed
    assert!(matches!(
        seektable_test(|blocks| {
            let seektable = blocks[1].clone();
            blocks.push(seektable);
        }),
        Err(Error::MultipleSeekTable)
    ));
}

#[test]
fn test_cuesheets() {
    use flac_codec::metadata::{Cuesheet, CuesheetError};

    assert!(Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/OK-0.cue"),).is_ok());
    assert!(Cuesheet::parse(44100 * 60 * 79, include_str!("data/cuesheets/OK-1.cue"),).is_ok());
    assert!(Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/OK-2.cue"),).is_ok());
    assert!(Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/OK-3.cue"),).is_ok());
    assert!(Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/OK-4.cue"),).is_ok());
    assert!(Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/OK-5.cue"),).is_ok());

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-MultipleCatalogNumber.cue")
        ),
        Err(CuesheetError::MultipleCatalogNumber)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-CatalogMissingNumber.cue")
        ),
        Err(CuesheetError::CatalogMissingNumber)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidCatalogNumber-1.cue")
        ),
        Err(CuesheetError::InvalidCatalogNumber)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidCatalogNumber-2.cue")
        ),
        Err(CuesheetError::InvalidCatalogNumber)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-PrematureFlags.cue")
        ),
        Err(CuesheetError::PrematureFlags)
    ));

    assert!(matches!(
        Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/BAD-LateFlags.cue")),
        Err(CuesheetError::LateFlags)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-PrematureIndex.cue")
        ),
        Err(CuesheetError::PrematureIndex)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidIndexPoint-1.cue")
        ),
        Err(CuesheetError::InvalidIndexPoint)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidIndexPoint-2.cue")
        ),
        Err(CuesheetError::InvalidIndexPoint)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidIndexPoint-3.cue")
        ),
        Err(CuesheetError::InvalidIndexPoint)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidIndexPoint-4.cue")
        ),
        Err(CuesheetError::InvalidIndexPoint)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidIndexPoint-5.cue")
        ),
        Err(CuesheetError::InvalidIndexPoint)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-IndexPointsOutOfSequence-1.cue")
        ),
        Err(CuesheetError::IndexPointsOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-IndexPointsOutOfSequence-2.cue")
        ),
        Err(CuesheetError::IndexPointsOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-IndexPointsOutOfSequence-3.cue")
        ),
        Err(CuesheetError::IndexPointsOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-IndexPointsOutOfSequence-4.cue")
        ),
        Err(CuesheetError::IndexPointsOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-MultipleISRC.cue")
        ),
        Err(CuesheetError::MultipleISRC)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-PrematureISRC.cue")
        ),
        Err(CuesheetError::PrematureISRC)
    ));

    assert!(matches!(
        Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/BAD-LateISRC.cue")),
        Err(CuesheetError::LateISRC)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidISRC-1.cue")
        ),
        Err(CuesheetError::InvalidISRC)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidISRC-2.cue")
        ),
        Err(CuesheetError::InvalidISRC)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-1.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-2.cue")
        ),
        Err(CuesheetError::IndexPointsOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-3.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-4.cue")
        ),
        Err(CuesheetError::IndexPointsOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-5.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-6.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-7.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-8.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-InvalidTrack-9.cue")
        ),
        Err(CuesheetError::InvalidTrack)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-TracksOutOfSequence-1.cue")
        ),
        Err(CuesheetError::TracksOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-TracksOutOfSequence-2.cue")
        ),
        Err(CuesheetError::TracksOutOfSequence)
    ));

    assert!(matches!(
        Cuesheet::parse(44100 * 10, include_str!("data/cuesheets/BAD-NoTracks.cue")),
        Err(CuesheetError::NoTracks)
    ));

    assert!(matches!(
        Cuesheet::parse(
            44100 * 10,
            include_str!("data/cuesheets/BAD-NonZeroFirstIndex.cue")
        ),
        Err(CuesheetError::NonZeroFirstIndex)
    ));
}

#[test]
fn test_block_position() {
    use flac_codec::metadata::{BlockList, BlockType, Padding, Streaminfo};

    let mut blocklist = BlockList::new(Streaminfo {
        minimum_block_size: 0,
        maximum_block_size: 0,
        minimum_frame_size: None,
        maximum_frame_size: None,
        sample_rate: 44100,
        channels: 1u8.try_into().unwrap(),
        bits_per_sample: 16u32.try_into().unwrap(),
        total_samples: None,
        md5: None,
    });

    // add comment
    let mut comment_1 = VorbisComment::default();
    comment_1.set("FOO", "bar");
    assert_eq!(blocklist.insert(comment_1.clone()), None);

    // add padding
    let padding = Padding { size: 10u8.into() };
    blocklist.insert(padding);

    // check order
    assert_eq!(
        blocklist
            .blocks()
            .map(|b| b.block_type())
            .collect::<Vec<_>>(),
        vec![
            BlockType::Streaminfo,
            BlockType::VorbisComment,
            BlockType::Padding
        ],
    );

    // add fresh comment
    let mut comment_2 = VorbisComment::default();
    comment_2.set("FOO", "baz");
    assert_eq!(blocklist.insert(comment_2), Some(comment_1));

    // order should be the same
    assert_eq!(
        blocklist
            .blocks()
            .map(|b| b.block_type())
            .collect::<Vec<_>>(),
        vec![
            BlockType::Streaminfo,
            BlockType::VorbisComment,
            BlockType::Padding
        ],
    );
}
