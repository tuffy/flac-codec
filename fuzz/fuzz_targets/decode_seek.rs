#![no_main]

use flac_codec::decode::{FlacSampleReader, Metadata};
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    if data.len() > 16 * 1024 * 1024 {
        return;
    }

    let Ok(mut reader) = FlacSampleReader::new_seekable(Cursor::new(data)) else {
        return;
    };
    let total_samples = reader.total_samples().unwrap_or_default();
    let derived = data
        .iter()
        .take(8)
        .enumerate()
        .fold(0u64, |value, (shift, byte)| {
            value | (u64::from(*byte) << (shift * 8))
        });

    for sample in [
        0,
        total_samples / 2,
        total_samples,
        total_samples.saturating_add(1),
        derived,
        u64::MAX,
    ] {
        if reader.seek(sample).is_ok() {
            let mut samples = [0; 64];
            let _ = reader.read(&mut samples);
        }
    }
});
