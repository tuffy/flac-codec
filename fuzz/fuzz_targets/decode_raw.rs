#![no_main]

use flac_codec::{
    decode::{FlacStreamReader, verify_reader},
    stream::FrameIterator,
};
use libfuzzer_sys::fuzz_target;
use std::io::BufReader;

fuzz_target!(|data: &[u8]| {
    if data.len() > 16 * 1024 * 1024 {
        return;
    }

    let _ = verify_reader(data);

    if let Ok(mut frames) = FrameIterator::new(data) {
        for _ in 0..16 {
            if frames.next().is_none() {
                break;
            }
        }
    }

    let mut stream = FlacStreamReader::new(BufReader::new(data));
    for _ in 0..4 {
        if stream.read().is_err() {
            break;
        }
    }
});
