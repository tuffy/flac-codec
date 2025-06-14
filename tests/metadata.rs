use flac_codec::Error;
use flac_codec::metadata::{
    Block, Cuesheet, CuesheetTrack, VorbisComment, read_blocks, write_blocks,
};

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

fn cuesheet_file_test(f: impl FnOnce(&mut Cuesheet)) -> Result<(), Error> {
    perform_test(
        include_bytes!("data/cuesheet.flac").as_slice(),
        |blocks| match &mut blocks[2] {
            Block::Cuesheet(cuesheet) => f(cuesheet),
            _ => panic!("cuesheet not found"),
        },
    )
}

fn cuesheet_test(f: impl FnOnce(&mut CuesheetTrack)) -> Result<(), Error> {
    cuesheet_file_test(|cuesheet| f(&mut cuesheet.tracks[1]))
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

    // SEEKTABLE points must be in proper order
    assert!(matches!(
        seektable_test(|blocks| {
            use flac_codec::metadata::SeekTable;

            match &mut blocks[1] {
                Block::SeekTable(SeekTable { points }) => {
                    points.swap(0, 1);
                }
                _ => panic!("seektable not found"),
            }
        }),
        Err(Error::InvalidSeekTablePoint)
    ));

    assert!(matches!(cuesheet_test(|_| { /* do nothing */ }), Ok(())));

    // the total number of CUESHEET tracks must fit into a u8
    assert!(matches!(
        cuesheet_file_test(|Cuesheet { tracks, .. }| {
            let track = tracks[0].clone();
            while tracks.len() < 256 {
                tracks.insert(0, track.clone());
            }
        }),
        Err(Error::ExcessiveCuesheetTracks)
    ));

    // CUESHEET index points must be evenly divisible by 588
    assert!(matches!(
        cuesheet_test(|track| {
            track.offset += 1;
        }),
        Err(Error::InvalidCuesheetOffset)
    ));

    // the total CUESHEET track index points must fit into a u8
    assert!(matches!(
        cuesheet_test(|track| {
            let point = track.index_points[0].clone();
            while track.index_points.len() < 256 {
                track.index_points.push(point.clone());
            }
        }),
        Err(Error::ExcessiveCuesheetIndexPoints)
    ));

    // non-lead out tracks must have at least 1 index point
    assert!(matches!(
        cuesheet_test(|track| {
            track.index_points.clear();
        }),
        Err(Error::InvalidCuesheetIndexPoints)
    ));

    // the first index point must be 0 or 1
    assert!(matches!(
        cuesheet_test(|track| {
            track.index_points[0].number = 2;
        }),
        Err(Error::InvalidCuesheetIndexPointNum)
    ));

    // index points must increment
    assert!(matches!(
        cuesheet_test(|track| {
            track.index_points[1].number = track.index_points[0].number;
        }),
        Err(Error::InvalidCuesheetIndexPointNum)
    ));

    // lead-out tracks must have no index points
    assert!(matches!(
        cuesheet_file_test(|Cuesheet { tracks, .. }| {
            use flac_codec::metadata::CuesheetIndexPoint;

            tracks
                .last_mut()
                .unwrap()
                .index_points
                .push(CuesheetIndexPoint {
                    number: 0,
                    offset: 0,
                });
        }),
        Err(Error::InvalidCuesheetIndexPoints)
    ));
}
