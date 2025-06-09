use flac_codec::{
    byteorder::LittleEndian,
    decode::{FlacReader, FlacSampleReader},
    encode::{EncodingOptions, FlacSampleWriter, FlacWriter},
};
use std::io::{Cursor, Read, Seek, Write};
use std::num::NonZero;

#[test]
fn test_small_files() {
    struct Samples {
        channels: NonZero<u8>,
        data: &'static [u8],
    }

    for Samples { channels, data } in [
        Samples {
            channels: NonZero::new(1).unwrap(),
            data: b"\x00\x80",
        },
        Samples {
            channels: NonZero::new(2).unwrap(),
            data: b"\x00\x80\xff\x7f",
        },
        Samples {
            channels: NonZero::new(1).unwrap(),
            data: b"\xe7\xff\x00\x00\x19\x00\x32\x00\x64\x00",
        },
        Samples {
            channels: NonZero::new(2).unwrap(),
            data:
                b"\xe7\xff\xf4\x01\x00\x00\x90\x01\x19\x00\x2c\x01\x32\x00\xc8\x00\x64\x00\x64\x00",
        },
    ] {
        let mut flac = Cursor::new(vec![]);
        let mut samples = data;

        assert_eq!(
            std::io::copy(
                &mut samples,
                &mut FlacWriter::endian(
                    &mut flac,
                    LittleEndian,
                    EncodingOptions::fast()
                        .max_lpc_order(NonZero::new(16))
                        .unwrap()
                        .mid_side(true)
                        .no_padding(),
                    44100,
                    16,
                    channels,
                    u64::try_from(data.len()).ok().and_then(NonZero::new),
                )
                .unwrap(),
            )
            .unwrap(),
            data.len().try_into().unwrap()
        );

        assert!(flac.rewind().is_ok());

        let mut output = vec![];

        assert_eq!(
            std::io::copy(
                &mut FlacReader::endian(flac, LittleEndian).unwrap(),
                &mut output,
            )
            .unwrap(),
            data.len().try_into().unwrap(),
        );

        assert_eq!(&output, &data);
    }
}

#[test]
fn test_blocksize_variations() {
    let data: &[u8] = include_bytes!("data/noise32.raw");

    for blocksize in [
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    ] {
        for lpc_order in &[0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32] {
            let mut flac = Cursor::new(vec![]);
            let mut samples = data;

            assert_eq!(
                std::io::copy(
                    &mut samples,
                    &mut FlacWriter::endian(
                        &mut flac,
                        LittleEndian,
                        EncodingOptions::best()
                            .max_lpc_order(NonZero::new(*lpc_order))
                            .unwrap()
                            .block_size(blocksize)
                            .unwrap()
                            .no_padding(),
                        44100,
                        8,
                        NonZero::new(1).unwrap(),
                        u64::try_from(data.len()).ok().and_then(NonZero::new),
                    )
                    .unwrap(),
                )
                .unwrap(),
                data.len().try_into().unwrap()
            );

            assert!(flac.rewind().is_ok());

            let mut output = vec![];

            assert_eq!(
                std::io::copy(
                    &mut FlacReader::endian(flac, LittleEndian).unwrap(),
                    &mut output,
                )
                .unwrap(),
                data.len().try_into().unwrap(),
            );

            assert_eq!(&output, &data);
        }
    }
}

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
                    u64::try_from(samples * 2 * 2).ok().and_then(NonZero::new),
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
    } in all
    {
        let mut flac = Cursor::new(vec![]);

        assert!(
            FlacWriter::endian(
                &mut flac,
                LittleEndian,
                EncodingOptions::default().no_padding(),
                44100,
                bps as u32,
                NonZero::new(channels as u8).unwrap(),
                u64::try_from(data.len()).ok().and_then(NonZero::new),
            )
            .unwrap()
            .write_all(data)
            .is_ok()
        );

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

#[test]
fn test_full_scale_deflection() {
    #[derive(Copy, Clone, Debug)]
    struct Samples {
        bps: u32,
        bytes: &'static [u8],
        iters: usize,
    }

    let all: [Samples; 28] = [
        Samples {
            bps: 8,
            bytes: b"\x7f\x80",
            iters: 100,
        },
        Samples {
            bps: 8,
            bytes: b"\x7f\x7f\x80",
            iters: 100,
        },
        Samples {
            bps: 8,
            bytes: b"\x7f\x80\x80",
            iters: 100,
        },
        Samples {
            bps: 8,
            bytes: b"\x7f\x80",
            iters: 200,
        },
        Samples {
            bps: 8,
            bytes: b"\x7f\x80\x80\x7f",
            iters: 200,
        },
        Samples {
            bps: 8,
            bytes: b"\x7f\x80\x7f\x7f\x80",
            iters: 100,
        },
        Samples {
            bps: 8,
            bytes: b"\x7f\x80\x80\x7f\x80",
            iters: 100,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\xff\x7f\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\x00\x80\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\x00\x80",
            iters: 200,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\x00\x80\x00\x80\xff\x7f",
            iters: 100,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\x00\x80\xff\x7f\xff\x7f\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 16,
            bytes: b"\xff\x7f\x00\x80\x00\x80\xff\x7f\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\xff\xff\x7f\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\x00\x00\x80\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\x00\x00\x80\xff\xff\x7f\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\x00\x00\x80\x00\x00\x80\xff\xff\x7f",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\x00\x00\x80\xff\xff\x7f\xff\xff\x7f\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 24,
            bytes: b"\xff\xff\x7f\x00\x00\x80\x00\x00\x80\xff\xff\x7f\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 32,
            bytes: b"\xff\xff\xff\x7f\x00\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 32,
            bytes: b"\xff\xff\xff\x7f\xff\xff\xff\x7f\x00\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 32,
            bytes: b"\xff\xff\xff\x7f\x00\x00\x00\x80\x00\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 32,
            bytes: b"\xff\xff\xff\x7f\x00\x00\x00\x80",
            iters: 200,
        },
        Samples {
            bps: 32,
            bytes: b"\xff\xff\xff\x7f\x00\x00\x00\x80\x00\x00\x00\x80\xff\xff\xff\x7f",
            iters: 100,
        },
        Samples {
            bps: 32,
            bytes:
                b"\xff\xff\xff\x7f\x00\x00\x00\x80\xff\xff\xff\x7f\xff\xff\xff\x7f\x00\x00\x00\x80",
            iters: 100,
        },
        Samples {
            bps: 32,
            bytes:
                b"\xff\xff\xff\x7f\x00\x00\x00\x80\x00\x00\x00\x80\xff\xff\xff\x7f\x00\x00\x00\x80",
            iters: 100,
        },
    ];

    for Samples { bps, bytes, iters } in all {
        let mut flac = Cursor::new(vec![]);

        let mut w = FlacWriter::endian(
            &mut flac,
            LittleEndian,
            EncodingOptions::default().no_padding(),
            44100,
            bps,
            NonZero::new(1).unwrap(),
            u64::try_from(bytes.len() * iters)
                .ok()
                .and_then(NonZero::new),
        )
        .unwrap();

        for _ in 0..iters {
            assert!(w.write_all(bytes).is_ok());
        }

        assert!(w.finalize().is_ok());
        assert!(flac.rewind().is_ok());

        let mut r = FlacReader::endian(flac, LittleEndian).unwrap();
        let mut buf = vec![];
        buf.resize(bytes.len(), 0);

        for _ in 0..iters {
            assert!(r.read_exact(&mut buf).is_ok());
            assert_eq!(buf.as_slice(), bytes);
        }
    }
}

#[test]
fn test_wasted_bits() {
    use flac_codec::{
        metadata::read_blocks,
        stream::{Frame, Subframe},
    };

    let data = include_bytes!("data/wasted-bits.raw");

    let mut flac = Cursor::new(vec![]);

    assert!(
        FlacWriter::endian(
            &mut flac,
            LittleEndian,
            EncodingOptions::default().no_padding(),
            44100,
            16,
            NonZero::new(1).unwrap(),
            u64::try_from(data.len()).ok().and_then(NonZero::new),
        )
        .unwrap()
        .write_all(data)
        .is_ok()
    );

    assert!(flac.rewind().is_ok());

    let mut output = vec![];

    // ensure file round-trips properly
    assert!(
        std::io::copy(
            &mut FlacReader::endian(&mut flac, LittleEndian).unwrap(),
            &mut output,
        )
        .is_ok(),
    );

    assert_eq!(&output, data);

    assert!(flac.rewind().is_ok());

    // ensure there are some actual wasted bits recorded
    assert!(read_blocks(&mut flac).count() > 0);
    let frame1 = Frame::read_subset(&mut flac).unwrap();
    assert!(
        match frame1.subframes[0] {
            Subframe::Constant { wasted_bps, .. }
            | Subframe::Verbatim { wasted_bps, .. }
            | Subframe::Fixed { wasted_bps, .. }
            | Subframe::Lpc { wasted_bps, .. } => wasted_bps,
        } > 0
    );
}

fn generate_sine_1(
    full_scale: f64,
    sample_rate: f64,
    samples: usize,
    f1: f64,
    a1: f64,
    f2: f64,
    a2: f64,
) -> Vec<i32> {
    use std::f64::consts::PI;

    let delta1: f64 = 2.0 * PI / (sample_rate / f1);
    let delta2: f64 = 2.0 * PI / (sample_rate / f2);
    let mut theta1: f64 = 0.0;
    let mut theta2: f64 = 0.0;

    (0..samples)
        .map(|_| {
            let val = a1 * theta1.sin() + a2 * theta2.sin() * full_scale;
            theta1 += delta1;
            theta2 += delta2;
            val as i32
        })
        .collect()
}

fn generate_sine_2(
    full_scale: f64,
    sample_rate: f64,
    samples: usize,
    f1: f64,
    a1: f64,
    f2: f64,
    a2: f64,
    fmult: f64,
) -> Vec<i32> {
    use std::f64::consts::PI;

    let delta1: f64 = 2.0 * PI / (sample_rate / f1);
    let delta2: f64 = 2.0 * PI / (sample_rate / f2);
    let mut theta1: f64 = 0.0;
    let mut theta2: f64 = 0.0;

    (0..samples)
        .map(|_| {
            let val = [
                a1 * theta1.sin() + a2 * theta2.sin() * full_scale,
                -(a1 * (theta1 * fmult).sin()) + a2 * (theta2 * fmult).sin() * full_scale,
            ];
            theta1 += delta1;
            theta2 += delta2;
            val.map(|v| v as i32)
        })
        .flatten()
        .collect()
}

#[test]
fn test_sine_wave_streams() {
    fn test_flac<const STEREO: bool, const SAMPLE_RATE: u32>(sine: Vec<i32>, bits_per_sample: u8) {
        let mut flac = Cursor::new(vec![]);

        let mut w = FlacSampleWriter::new(
            &mut flac,
            EncodingOptions::default(),
            SAMPLE_RATE,
            u32::from(bits_per_sample),
            match STEREO {
                true => NonZero::new(2).unwrap(),
                false => NonZero::new(1).unwrap(),
            },
            sine.len().try_into().ok().and_then(NonZero::new),
        )
        .unwrap();

        assert!(w.write(&sine).is_ok());
        assert!(w.finalize().is_ok());

        assert!(flac.rewind().is_ok());

        let mut r = FlacSampleReader::new(flac).unwrap();
        let mut sine = sine.as_slice();

        loop {
            match r.fill_buf() {
                Ok([]) => break,
                Ok(buf) => {
                    let (start, rest) = sine.split_at(buf.len());
                    assert_eq!(buf, start);
                    r.consume(start.len());
                    sine = rest;
                }
                Err(_) => panic!("error reading from FLAC file"),
            }
        }
    }

    for bits_per_sample in [8, 16, 24, 32] {
        test_flac::<false, 48000>(
            // sine{BPS}-00
            generate_sine_1(
                f64::from(1 << (bits_per_sample - 1)),
                48000.0,
                200000,
                441.0,
                0.50,
                441.0,
                0.49,
            ),
            bits_per_sample,
        );

        test_flac::<false, 96000>(
            // sine{BPS}-01
            generate_sine_1(
                f64::from(1 << (bits_per_sample - 1)),
                96000.0,
                200000,
                441.0,
                0.61,
                661.5,
                0.37,
            ),
            bits_per_sample,
        );

        for sine in [
            // sine{BPS}-02
            generate_sine_1(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.50,
                882.0,
                0.49,
            ),
            // sine{BPS}-03
            generate_sine_1(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.50,
                4410.0,
                0.49,
            ),
            // sine{BPS}-04
            generate_sine_1(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                8820.0,
                0.70,
                4410.0,
                0.29,
            ),
        ] {
            test_flac::<false, 44100>(sine, bits_per_sample);
        }

        for sine in [
            // sine{BPS}-10
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                48000.0,
                200000,
                441.0,
                0.50,
                441.0,
                0.49,
                1.0,
            ),
            // sine{BPS}-11
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                48000.0,
                200000,
                441.0,
                0.61,
                661.5,
                0.37,
                1.0,
            ),
        ] {
            test_flac::<true, 48000>(sine, bits_per_sample);
        }

        // sine{BPS}-12
        test_flac::<true, 96000>(
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                96000.0,
                200000,
                441.0,
                0.50,
                882.0,
                0.49,
                1.0,
            ),
            bits_per_sample,
        );

        for sine in [
            // sine{BPS}-13
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.50,
                4410.0,
                0.49,
                1.0,
            ),
            // sine{BPS}-14
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                8820.0,
                0.70,
                4410.0,
                0.29,
                1.0,
            ),
            // sine{BPS}-15
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.50,
                441.0,
                0.49,
                0.5,
            ),
            // sine{BPS}-16
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.61,
                661.5,
                0.37,
                2.0,
            ),
            // sine{BPS}-17
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.50,
                882.0,
                0.49,
                0.7,
            ),
            // sine{BPS}-18
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                441.0,
                0.50,
                4410.0,
                0.49,
                1.3,
            ),
            // sine{BPS}-19
            generate_sine_2(
                f64::from(1 << (bits_per_sample - 1)),
                44100.0,
                200000,
                8820.0,
                0.70,
                4410.0,
                0.29,
                0.1,
            ),
        ] {
            test_flac::<true, 44100>(sine, bits_per_sample);
        }
    }
}

#[test]
fn test_noise() {
    enum Opt {
        Fast,
        Best,
    }

    let noise = std::iter::repeat_with(|| fastrand::u8(..))
        .take(1572864)
        .collect::<Vec<u8>>();

    for channels in [1, 2, 4, 8] {
        for bits_per_sample in [8, 16, 24, 32] {
            for option in [None, Some(Opt::Fast), Some(Opt::Best)] {
                for block_size in [None, Some(32), Some(32768), Some(65535)] {
                    let mut flac = Cursor::new(vec![]);

                    assert!(
                        FlacWriter::endian(
                            &mut flac,
                            LittleEndian,
                            {
                                let mut opt = match option {
                                    None => EncodingOptions::default(),
                                    Some(Opt::Fast) => EncodingOptions::fast(),
                                    Some(Opt::Best) => EncodingOptions::best(),
                                };
                                opt = match block_size {
                                    None => opt,
                                    Some(size) => opt.block_size(size).unwrap(),
                                };
                                opt.no_padding()
                            },
                            44100,
                            bits_per_sample,
                            NonZero::new(channels).unwrap(),
                            u64::try_from(noise.len()).ok().and_then(NonZero::new),
                        )
                        .unwrap()
                        .write_all(&noise)
                        .is_ok()
                    );

                    assert!(flac.rewind().is_ok());

                    let mut output = vec![];

                    // ensure file round-trips properly
                    assert!(
                        std::io::copy(
                            &mut FlacReader::endian(&mut flac, LittleEndian).unwrap(),
                            &mut output,
                        )
                        .is_ok(),
                    );

                    assert_eq!(&output, &noise);
                }
            }
        }
    }
}
