#[test]
fn test_fractional() {
    fn perform_test(blocksize: u16, samples: usize) {
        use flac_codec::{
            byteorder::LittleEndian,
            decode::FlacReader,
            encode::{EncodingOptions, FlacWriter},
        };
        use std::io::{Cursor, Seek};
        use std::num::NonZero;

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
