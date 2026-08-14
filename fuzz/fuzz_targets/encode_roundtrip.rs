#![no_main]

use flac_codec::{
    decode::FlacSampleReader,
    encode::{FlacSampleWriter, Options},
};
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fn sample_for_bps(value: u32, bits_per_sample: u32) -> i32 {
    let shift = 32 - bits_per_sample;
    (value << shift) as i32 >> shift
}

fuzz_target!(|data: &[u8]| {
    let Some((&bits, data)) = data.split_first() else {
        return;
    };
    let Some((&channel_count, data)) = data.split_first() else {
        return;
    };
    let Some((&option, data)) = data.split_first() else {
        return;
    };
    let Some((&block_size, data)) = data.split_first() else {
        return;
    };

    let bits_per_sample = 1 + u32::from(bits % 32);
    let channels = 1 + channel_count % 8;
    let block_size = 16 + u16::from(block_size);
    let pcm_frames = (data.len() / 4 / usize::from(channels)).min(128);
    if pcm_frames == 0 {
        return;
    }

    let samples = data
        .chunks_exact(4)
        .take(pcm_frames * usize::from(channels))
        .map(|bytes| {
            sample_for_bps(
                u32::from_le_bytes(bytes.try_into().unwrap()),
                bits_per_sample,
            )
        })
        .collect::<Vec<_>>();
    let options = match option % 3 {
        0 => Options::fast(),
        1 => Options::default(),
        _ => Options::best(),
    }
    .block_size(block_size)
    .unwrap()
    .no_padding();

    let mut encoded = Cursor::new(Vec::new());
    let mut writer = FlacSampleWriter::new(
        &mut encoded,
        options,
        44_100,
        bits_per_sample,
        channels,
        Some(samples.len() as u64),
    )
    .unwrap();
    writer.write(&samples).unwrap();
    writer.finalize().unwrap();

    let flac = encoded.into_inner();
    let mut decoded = Vec::new();
    FlacSampleReader::new(flac.as_slice())
        .unwrap()
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, samples);
});
