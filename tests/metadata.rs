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
}
