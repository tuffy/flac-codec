#[test]
fn test_byte_seeking() {
    use flac_codec::byteorder::LittleEndian;
    use flac_codec::decode::{FlacByteReader, Metadata};
    use std::io::{Cursor, Read, Seek, SeekFrom};

    let mut flac: FlacByteReader<_, LittleEndian> =
        FlacByteReader::new_seekable(Cursor::new(include_bytes!("data/sine.flac"))).unwrap();

    let flac_len = flac.total_samples().unwrap();

    let bytes_per_pcm_frame =
        u64::from(flac.channel_count()) * u64::from(u32::from(flac.bits_per_sample()) / 8);

    let mut all_data = Vec::new();

    assert!(flac.read_to_end(&mut all_data).is_ok());

    assert_eq!(all_data.len() as u64, flac_len * bytes_per_pcm_frame);

    let mut md5 = md5::Context::new();
    md5.consume(&all_data);
    assert_eq!(&md5.compute().0, flac.md5().unwrap());

    // test seeking from start
    for i in 1..(flac_len / 10000) {
        assert!(
            flac.seek(SeekFrom::Start(i * 10000 * bytes_per_pcm_frame))
                .is_ok()
        );
        let mut rest = Vec::new();
        assert!(flac.read_to_end(&mut rest).is_ok());
        assert_eq!(
            rest.len() as u64,
            (flac_len * bytes_per_pcm_frame) - (i * 10000 * bytes_per_pcm_frame)
        );
        assert!(all_data.ends_with(&rest));
    }

    // test seeking from current position
    assert!(flac.rewind().is_ok());
    let mut chunk1 = [0; 10000];
    assert!(flac.read_exact(&mut chunk1).is_ok());
    let mut chunk2 = [0; 10000];
    assert!(flac.read_exact(&mut chunk2).is_ok());
    assert!(flac.seek(SeekFrom::Current(-10000)).is_ok());
    let mut chunk3 = [0; 10000];
    assert!(flac.read_exact(&mut chunk3).is_ok());
    assert_eq!(chunk2, chunk3);
    assert!(flac.rewind().is_ok());
    assert!(flac.seek(SeekFrom::Current(10000)).is_ok());
    assert!(flac.read_exact(&mut chunk3).is_ok());
    assert_eq!(chunk2, chunk3);

    // test seeking from end
    assert!(flac.seek(SeekFrom::End(-10000)).is_ok());
    let mut chunk1 = [0; 10000];
    assert!(flac.read_exact(&mut chunk1).is_ok());
    assert!(all_data.ends_with(&chunk1));

    assert!(flac.seek(SeekFrom::End(-20000)).is_ok());
    let mut chunk1 = [0; 20000];
    assert!(flac.read_exact(&mut chunk1).is_ok());
    assert!(all_data.ends_with(&chunk1));

    assert!(flac.seek(SeekFrom::End(-30000)).is_ok());
    let mut chunk1 = [0; 30000];
    assert!(flac.read_exact(&mut chunk1).is_ok());
    assert!(all_data.ends_with(&chunk1));

    // test some invalid seeks
    assert!(flac.rewind().is_ok());
    assert!(
        flac.seek(SeekFrom::Start(flac_len * bytes_per_pcm_frame + 1))
            .is_err()
    );
    assert!(flac.seek(SeekFrom::Start(u64::MAX)).is_err());

    assert!(
        flac.seek(SeekFrom::Current(
            (flac_len * bytes_per_pcm_frame + 1).try_into().unwrap()
        ))
        .is_err()
    );
    assert!(flac.seek(SeekFrom::Current(i64::MAX)).is_err());
    assert!(flac.seek(SeekFrom::Current(i64::MIN)).is_err());

    assert!(flac.seek(SeekFrom::End(i64::MIN)).is_err());
    assert!(flac.seek(SeekFrom::End(1)).is_err());
}

#[test]
fn test_sample_seeking() {
    use flac_codec::decode::{FlacSampleRead, Metadata, SeekableFlacSampleReader};
    use std::io::Cursor;

    let mut flac =
        SeekableFlacSampleReader::new(Cursor::new(include_bytes!("data/sine.flac"))).unwrap();

    assert_eq!(flac.channel_count(), 2);

    // FLAC length in channel-independent samples
    let flac_len = flac.total_samples().unwrap();
    assert_eq!(flac_len, 200000);

    let mut all_data = Vec::new();

    assert!(flac.read_to_end(&mut all_data).is_ok());
    assert_eq!(all_data.len(), 200000 * 2);

    // this trait only supports seeking from the start
    for i in 1..(flac_len / 10000) {
        assert!(flac.seek(i * 10000).is_ok());
        let mut rest = Vec::new();
        assert!(flac.read_to_end(&mut rest).is_ok());
        assert_eq!(rest.len() as u64, (flac_len * 2) - (i * 10000 * 2));
        assert!(all_data.ends_with(&rest));
    }

    // test an invalid seek or two
    assert!(flac.seek(200000).is_ok());
    assert!(flac.seek(200001).is_err());
    assert!(flac.seek(u64::MAX).is_err());
}
