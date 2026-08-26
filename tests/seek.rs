// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_byte_seeking() {
    use flac_codec::byteorder::LittleEndian;
    use flac_codec::decode::{FlacByteReader, Metadata};
    use std::io::{Cursor, Read, Seek, SeekFrom};

    let mut flac: FlacByteReader<_, LittleEndian> =
        FlacByteReader::new_seekable(Cursor::new(include_bytes!("data/sine.flac"))).unwrap();

    let flac_len = flac.total_samples().unwrap();

    let bytes_per_pcm_frame =
        u64::from(flac.channel_count()) * u64::from(flac.bits_per_sample() / 8);

    let mut all_data = Vec::new();

    assert!(flac.read_to_end(&mut all_data).is_ok());

    assert_eq!(all_data.len() as u64, flac_len * bytes_per_pcm_frame);

    let mut md5 = md5::Context::new();
    md5.consume(&all_data);
    assert_eq!(&md5.finalize().0, flac.md5().unwrap());

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
    use flac_codec::decode::{FlacSampleReader, Metadata};
    use std::io::Cursor;

    let mut flac =
        FlacSampleReader::new_seekable(Cursor::new(include_bytes!("data/sine.flac"))).unwrap();

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

#[test]
fn test_failed_sample_seek_preserves_decoder_state() {
    use flac_codec::decode::{FlacSampleReader, Metadata};
    use std::io::Cursor;

    let mut flac =
        FlacSampleReader::new_seekable(Cursor::new(include_bytes!("data/sine.flac"))).unwrap();
    let total_samples = flac.total_samples().unwrap();

    assert!(flac.seek(total_samples + 1).is_err());
    assert!(flac.seek(0).is_ok());

    let mut samples = [0; 32];
    assert_eq!(flac.read(&mut samples).unwrap(), samples.len());
}

#[test]
fn test_byte_seek_from_end_uses_byte_length() {
    use flac_codec::{
        byteorder::LittleEndian,
        decode::FlacByteReader,
        encode::{FlacSampleWriter, Options},
    };
    use std::io::{Cursor, Read, Seek, SeekFrom};

    let samples = (0..101)
        .flat_map(|sample| [sample * 197 - 10_000, sample * -131 + 7_000])
        .collect::<Vec<_>>();
    let mut encoded = Cursor::new(Vec::new());
    let mut writer = FlacSampleWriter::new(
        &mut encoded,
        Options::fast().block_size(16).unwrap().no_padding(),
        44_100,
        16,
        2,
        Some(samples.len() as u64),
    )
    .unwrap();
    writer.write(&samples).unwrap();
    writer.finalize().unwrap();

    let mut reader =
        FlacByteReader::<_, LittleEndian>::new_seekable(Cursor::new(encoded.into_inner())).unwrap();
    let mut all_bytes = Vec::new();
    reader.read_to_end(&mut all_bytes).unwrap();

    assert_eq!(
        reader.seek(SeekFrom::End(-7)).unwrap(),
        all_bytes.len() as u64 - 7
    );
    let mut suffix = Vec::new();
    reader.read_to_end(&mut suffix).unwrap();
    assert_eq!(suffix, all_bytes[all_bytes.len() - 7..]);
}
