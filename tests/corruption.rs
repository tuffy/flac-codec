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

    for _ in 0..100 {
        let mut flac: Vec<u8> = Vec::from(flac);
        flac[fastrand::usize(valid_range.clone())] ^= 1 << fastrand::u32(0..8);

        assert!(
            copy(
                &mut FlacByteReader::endian(&flac[..], LittleEndian).unwrap(),
                &mut sink()
            )
            .is_err()
        );
    }
}
