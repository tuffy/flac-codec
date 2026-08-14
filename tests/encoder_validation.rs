use flac_codec::{
    Error,
    byteorder::LittleEndian,
    decode::{FlacByteReader, FlacSampleReader},
    encode::{FlacByteWriter, FlacChannelWriter, FlacSampleWriter, FlacStreamWriter, Options},
};
use std::io::{Cursor, Read, Write};

fn sample_for_bps(value: u32, bits_per_sample: u32) -> i32 {
    let shift = 32 - bits_per_sample;
    (value << shift) as i32 >> shift
}

#[test]
fn roundtrip_non_byte_aligned_byte_input() {
    for bits_per_sample in [1u32, 7, 9, 17, 25, 31] {
        for channels in [1u8, 8] {
            let bytes_per_sample = bits_per_sample.div_ceil(8) as usize;
            let mut input = Vec::new();
            for pcm_frame in 0..35u32 {
                for channel in 0..channels {
                    let sample = sample_for_bps(
                        pcm_frame.wrapping_mul(0x9E37_79B9) ^ u32::from(channel),
                        bits_per_sample,
                    );
                    input.extend_from_slice(&sample.to_le_bytes()[..bytes_per_sample]);
                }
            }

            let mut encoded = Cursor::new(Vec::new());
            let mut writer = FlacByteWriter::endian(
                &mut encoded,
                LittleEndian,
                Options::fast().block_size(16).unwrap().no_padding(),
                44_100,
                bits_per_sample,
                channels,
                Some(input.len() as u64),
            )
            .unwrap();
            writer.write_all(&input).unwrap();
            writer.finalize().unwrap();

            let mut decoded = Vec::new();
            FlacByteReader::endian(encoded.into_inner().as_slice(), LittleEndian)
                .unwrap()
                .read_to_end(&mut decoded)
                .unwrap();
            assert_eq!(decoded, input, "{bits_per_sample} bps, {channels} channels");
        }
    }
}

#[test]
fn reject_invalid_encoder_parameters_without_panicking() {
    let mut encoded = Cursor::new(Vec::new());
    assert!(matches!(
        FlacSampleWriter::new(&mut encoded, Options::fast(), 44_100, 16, 0, Some(1)),
        Err(Error::ExcessiveChannels)
    ));
    assert!(matches!(
        FlacByteWriter::<_, LittleEndian>::new(
            &mut encoded,
            Options::fast(),
            44_100,
            16,
            0,
            Some(1)
        ),
        Err(Error::ExcessiveChannels)
    ));
    assert!(matches!(
        FlacChannelWriter::new(&mut encoded, Options::fast(), 44_100, 16, 0, Some(1)),
        Err(Error::ExcessiveChannels)
    ));
    assert!(matches!(
        FlacSampleWriter::new(&mut encoded, Options::fast(), 44_100, 0, 1, None),
        Err(Error::InvalidBitsPerSample)
    ));
    assert!(matches!(
        FlacSampleWriter::new(&mut encoded, Options::fast(), 44_100, 33, 1, None),
        Err(Error::InvalidBitsPerSample)
    ));
    assert!(matches!(
        FlacSampleWriter::new(&mut encoded, Options::fast(), 1 << 20, 16, 1, None),
        Err(Error::InvalidSampleRate)
    ));

    let mut stream = FlacStreamWriter::new(Vec::new(), Options::fast());
    assert!(matches!(
        stream.write(44_100, 0, 16, &[]),
        Err(Error::ExcessiveChannels)
    ));
}

fn roundtrip_options() -> [Options; 2] {
    [
        Options::fast().block_size(16).unwrap().no_padding(),
        Options::best().block_size(17).unwrap().no_padding(),
    ]
}

#[test]
fn roundtrip_every_bit_depth_and_channel_extreme() {
    for bits_per_sample in 1..=32 {
        for channels in [1u8, 2, 8] {
            let mut state = 0x1234_5678u32 ^ bits_per_sample ^ u32::from(channels);
            let mut samples = (0..(37 * usize::from(channels)))
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    sample_for_bps(state, bits_per_sample)
                })
                .collect::<Vec<_>>();

            samples[0] = if bits_per_sample == 32 {
                i32::MIN
            } else {
                -(1 << (bits_per_sample - 1))
            };
            samples[1] = if bits_per_sample == 32 {
                i32::MAX
            } else {
                (1 << (bits_per_sample - 1)) - 1
            };

            for options in roundtrip_options() {
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

                let mut decoded = Vec::new();
                FlacSampleReader::new(encoded.into_inner().as_slice())
                    .unwrap()
                    .read_to_end(&mut decoded)
                    .unwrap();
                assert_eq!(
                    decoded, samples,
                    "{bits_per_sample} bps, {channels} channels"
                );
            }
        }
    }
}

#[test]
fn roundtrip_short_fixed_predictor_frames() {
    let samples = [-14_336, -65_536, -1, 65_535];

    for sample_count in 1..=samples.len() {
        let samples = &samples[..sample_count];
        let mut encoded = Cursor::new(Vec::new());
        let mut writer = FlacSampleWriter::new(
            &mut encoded,
            Options::fast().block_size(76).unwrap().no_padding(),
            44_100,
            24,
            1,
            Some(samples.len() as u64),
        )
        .unwrap();
        writer.write(samples).unwrap();
        writer.finalize().unwrap();

        let mut decoded = Vec::new();
        FlacSampleReader::new(encoded.into_inner().as_slice())
            .unwrap()
            .read_to_end(&mut decoded)
            .unwrap();
        assert_eq!(decoded, samples);
    }
}

#[test]
fn reject_partial_interleaved_sample_frame() {
    let mut encoded = Cursor::new(Vec::new());
    let mut writer =
        FlacSampleWriter::new(&mut encoded, Options::fast(), 44_100, 16, 2, None).unwrap();
    writer.write(&[1, 2, 3]).unwrap();

    assert!(matches!(
        writer.finalize(),
        Err(Error::SamplesNotDivisibleByChannels)
    ));
}

#[test]
fn reject_partial_pcm_byte_frame() {
    let mut encoded = Cursor::new(Vec::new());
    let mut writer = FlacByteWriter::endian(
        &mut encoded,
        LittleEndian,
        Options::fast(),
        44_100,
        16,
        2,
        None,
    )
    .unwrap();
    writer.write_all(&[1, 2, 3]).unwrap();

    assert!(matches!(
        writer.finalize(),
        Err(Error::SamplesNotDivisibleByChannels)
    ));
}

#[test]
fn reject_samples_outside_declared_bit_depth() {
    let mut encoded = Cursor::new(Vec::new());
    let mut writer = FlacSampleWriter::new(
        &mut encoded,
        Options::fast().block_size(16).unwrap(),
        44_100,
        4,
        2,
        Some(32),
    )
    .unwrap();

    assert!(matches!(
        writer.write(&[i32::MAX, i32::MIN].repeat(16)),
        Err(Error::SampleOutOfRange)
    ));
}
