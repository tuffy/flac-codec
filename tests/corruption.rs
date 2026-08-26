// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_file_corruption() {
    use flac_codec::byteorder::LittleEndian;
    use flac_codec::decode::FlacByteReader;
    use std::io::{copy, sink};

    // ensure test file is okay
    let flac = include_bytes!("data/sine.flac");

    assert!(
        copy(
            &mut FlacByteReader::endian(&flac[..], LittleEndian).unwrap(),
            &mut sink()
        )
        .is_ok()
    );

    // try swapping some random bits outside the metadata block area
    // (flipping a bit somewhere in a PADDING block might not be
    // noticed, for instance)
    let valid_range = 136..flac.len();
    let mut rng = fastrand::Rng::with_seed(0x7E57_C0DE);

    for _ in 0..100 {
        let mut flac: Vec<u8> = Vec::from(flac);
        let byte = rng.usize(valid_range.clone());
        let bit = rng.u32(0..8);
        flac[byte] ^= 1 << bit;

        assert!(
            copy(
                &mut FlacByteReader::endian(&flac[..], LittleEndian).unwrap(),
                &mut sink()
            )
            .is_err(),
            "corruption at byte {byte}, bit {bit} was not detected"
        );
    }
}

#[test]
fn test_every_single_frame_bit_corruption() {
    use flac_codec::{
        byteorder::LittleEndian,
        decode::FlacByteReader,
        encode::{FlacSampleWriter, Options},
        metadata::BlockList,
    };
    use std::io::{Cursor, copy, sink};

    let mut flac = Cursor::new(Vec::new());
    let mut writer = FlacSampleWriter::new(
        &mut flac,
        Options::fast().block_size(16).unwrap().no_padding(),
        44100,
        16,
        1,
        Some(33),
    )
    .unwrap();
    writer.write(&(0..33).collect::<Vec<_>>()).unwrap();
    writer.finalize().unwrap();

    let mut flac = flac.into_inner();
    let mut input = Cursor::new(flac.as_slice());
    BlockList::read(&mut input).unwrap();
    let frames_start = usize::try_from(input.position()).unwrap();

    for byte in frames_start..flac.len() {
        for bit in 0..8 {
            flac[byte] ^= 1 << bit;
            let result = FlacByteReader::endian(flac.as_slice(), LittleEndian)
                .and_then(|mut reader| copy(&mut reader, &mut sink()).map_err(Into::into));
            flac[byte] ^= 1 << bit;

            assert!(
                result.is_err(),
                "corruption at byte {byte}, bit {bit} was not detected"
            );
        }
    }
}
