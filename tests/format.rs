use flac_codec::{
    byteorder::LittleEndian,
    decode::FlacReader,
    encode::{EncodingOptions, FlacWriter},
};
use std::io::{Cursor, Seek, Write};
use std::num::NonZero;

#[test]
fn test_fractional() {
    fn perform_test(blocksize: u16, samples: usize) {
        let data = include_bytes!("data/noise.raw");
        let mut slice = &data[0..samples * 4];

        let mut flac = Cursor::new(vec![]);

        assert_eq!(
            std::io::copy(
                &mut slice,
                &mut FlacWriter::endian(
                    &mut flac,
                    LittleEndian,
                    EncodingOptions::default()
                        .block_size(blocksize)
                        .unwrap()
                        .no_padding(),
                    44100,
                    16,
                    NonZero::new(2).unwrap(),
                    u64::try_from(samples).ok().and_then(NonZero::new),
                )
                .unwrap(),
            )
            .unwrap(),
            (samples * 4).try_into().unwrap()
        );

        assert!(flac.rewind().is_ok());

        let mut output = vec![];

        assert_eq!(
            std::io::copy(
                &mut FlacReader::endian(flac, LittleEndian).unwrap(),
                &mut output,
            )
            .unwrap(),
            (samples * 4).try_into().unwrap(),
        );

        assert_eq!(&output, &data[0..samples * 4]);
    }

    for samples in [31, 32, 33, 34, 35, 2046, 2047, 2048, 2049, 2050] {
        perform_test(33, samples);
    }

    for samples in [
        254, 255, 256, 257, 258, 510, 511, 512, 513, 514, 1022, 1023, 1024, 1025, 1026, 2046, 2047,
        2048, 2049, 2050, 4094, 4095, 4096, 4097, 4098,
    ] {
        perform_test(256, samples);
    }

    for samples in [
        1022, 1023, 1024, 1025, 1026, 2046, 2047, 2048, 2049, 2050, 4094, 4095, 4096, 4097, 4098,
    ] {
        perform_test(2048, samples);
    }

    for samples in [
        1022, 1023, 1024, 1025, 1026, 2046, 2047, 2048, 2049, 2050, 4094, 4095, 4096, 4097, 4098,
        4606, 4607, 4608, 4609, 4610, 8190, 8191, 8192, 8193, 8194, 16382, 16383, 16384, 16385,
        16386,
    ] {
        perform_test(4608, samples);
    }
}

#[test]
fn test_roundtrip() {
    struct Samples {
        channels: usize,
        bps: usize,
        data: &'static [u8],
    }

    let all: [Samples; 36] = [
        Samples {
            channels: 1,
            bps: 8,
            data: include_bytes!("data/roundtrip-1-8-1.raw"),
        },
        Samples {
            channels: 2,
            bps: 8,
            data: include_bytes!("data/roundtrip-2-8-1.raw"),
        },
        Samples {
            channels: 4,
            bps: 8,
            data: include_bytes!("data/roundtrip-4-8-1.raw"),
        },
        Samples {
            channels: 8,
            bps: 8,
            data: include_bytes!("data/roundtrip-8-8-1.raw"),
        },
        Samples {
            channels: 1,
            bps: 8,
            data: include_bytes!("data/roundtrip-1-8-111.raw"),
        },
        Samples {
            channels: 2,
            bps: 8,
            data: include_bytes!("data/roundtrip-2-8-111.raw"),
        },
        Samples {
            channels: 4,
            bps: 8,
            data: include_bytes!("data/roundtrip-4-8-111.raw"),
        },
        Samples {
            channels: 8,
            bps: 8,
            data: include_bytes!("data/roundtrip-8-8-111.raw"),
        },
        Samples {
            channels: 1,
            bps: 8,
            data: include_bytes!("data/roundtrip-1-8-4777.raw"),
        },
        Samples {
            channels: 2,
            bps: 8,
            data: include_bytes!("data/roundtrip-2-8-4777.raw"),
        },
        Samples {
            channels: 4,
            bps: 8,
            data: include_bytes!("data/roundtrip-4-8-4777.raw"),
        },
        Samples {
            channels: 8,
            bps: 8,
            data: include_bytes!("data/roundtrip-8-8-4777.raw"),
        },
        Samples {
            channels: 1,
            bps: 16,
            data: include_bytes!("data/roundtrip-1-16-1.raw"),
        },
        Samples {
            channels: 2,
            bps: 16,
            data: include_bytes!("data/roundtrip-2-16-1.raw"),
        },
        Samples {
            channels: 4,
            bps: 16,
            data: include_bytes!("data/roundtrip-4-16-1.raw"),
        },
        Samples {
            channels: 8,
            bps: 16,
            data: include_bytes!("data/roundtrip-8-16-1.raw"),
        },
        Samples {
            channels: 1,
            bps: 16,
            data: include_bytes!("data/roundtrip-1-16-111.raw"),
        },
        Samples {
            channels: 2,
            bps: 16,
            data: include_bytes!("data/roundtrip-2-16-111.raw"),
        },
        Samples {
            channels: 4,
            bps: 16,
            data: include_bytes!("data/roundtrip-4-16-111.raw"),
        },
        Samples {
            channels: 8,
            bps: 16,
            data: include_bytes!("data/roundtrip-8-16-111.raw"),
        },
        Samples {
            channels: 1,
            bps: 16,
            data: include_bytes!("data/roundtrip-1-16-4777.raw"),
        },
        Samples {
            channels: 2,
            bps: 16,
            data: include_bytes!("data/roundtrip-2-16-4777.raw"),
        },
        Samples {
            channels: 4,
            bps: 16,
            data: include_bytes!("data/roundtrip-4-16-4777.raw"),
        },
        Samples {
            channels: 8,
            bps: 16,
            data: include_bytes!("data/roundtrip-8-16-4777.raw"),
        },
        Samples {
            channels: 1,
            bps: 24,
            data: include_bytes!("data/roundtrip-1-24-1.raw"),
        },
        Samples {
            channels: 2,
            bps: 24,
            data: include_bytes!("data/roundtrip-2-24-1.raw"),
        },
        Samples {
            channels: 4,
            bps: 24,
            data: include_bytes!("data/roundtrip-4-24-1.raw"),
        },
        Samples {
            channels: 8,
            bps: 24,
            data: include_bytes!("data/roundtrip-8-24-1.raw"),
        },
        Samples {
            channels: 1,
            bps: 24,
            data: include_bytes!("data/roundtrip-1-24-111.raw"),
        },
        Samples {
            channels: 2,
            bps: 24,
            data: include_bytes!("data/roundtrip-2-24-111.raw"),
        },
        Samples {
            channels: 4,
            bps: 24,
            data: include_bytes!("data/roundtrip-4-24-111.raw"),
        },
        Samples {
            channels: 8,
            bps: 24,
            data: include_bytes!("data/roundtrip-8-24-111.raw"),
        },
        Samples {
            channels: 1,
            bps: 24,
            data: include_bytes!("data/roundtrip-1-24-4777.raw"),
        },
        Samples {
            channels: 2,
            bps: 24,
            data: include_bytes!("data/roundtrip-2-24-4777.raw"),
        },
        Samples {
            channels: 4,
            bps: 24,
            data: include_bytes!("data/roundtrip-4-24-4777.raw"),
        },
        Samples {
            channels: 8,
            bps: 24,
            data: include_bytes!("data/roundtrip-8-24-4777.raw"),
        },
    ];

    for Samples {
        channels,
        bps,
        data,
    } in all {
        let mut flac = Cursor::new(vec![]);

        assert!(FlacWriter::endian(
            &mut flac,
            LittleEndian,
            EncodingOptions::default().no_padding(),
            44100,
            bps as u32,
            NonZero::new(channels as u8).unwrap(),
            u64::try_from(data.len() / (bps / 8) / channels).ok().and_then(NonZero::new),
        ).unwrap().write_all(data).is_ok());

        assert!(flac.rewind().is_ok());

        let mut output = vec![];

        assert!(
            std::io::copy(
                &mut FlacReader::endian(flac, LittleEndian).unwrap(),
                &mut output,
            )
            .is_ok(),
        );

        assert_eq!(&output, data);
    }
}
