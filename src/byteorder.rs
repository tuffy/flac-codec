// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Byte order for PCM samples

/// Sample byte order
pub trait Endianness {
    /// Converts 8-bit sample to bytes in this byte order
    fn i8_to_bytes(sample: i8) -> [u8; 1];

    /// Converts 16-bit sample to bytes in this byte order
    fn i16_to_bytes(sample: i16) -> [u8; 2];

    /// Converts 24-bit sample to bytes in this byte order
    fn i24_to_bytes(sample: i32) -> [u8; 3];

    /// Converts 32-bit sample to bytes in this byte order
    fn i32_to_bytes(sample: i32) -> [u8; 4];

    /// Converts bytes to 8-bit samples in this byte order
    fn bytes_to_i8(bytes: [u8; 1]) -> i8;

    /// Converts bytes to 16-bit samples in this byte order
    fn bytes_to_i16(bytes: [u8; 2]) -> i16;

    /// Converts bytes to 24-bit samples in this byte order
    fn bytes_to_i24(bytes: [u8; 3]) -> i32;

    /// Converts bytes to 32-bit samples in this byte order
    fn bytes_to_i32(bytes: [u8; 4]) -> i32;

    /// Converts bytes in this byte order to big-endian
    fn bytes_to_be(buf: &mut [u8], bytes_per_sample: usize);

    /// Converts bytes in this byte order to little-endian
    fn bytes_to_le(buf: &mut [u8], bytes_per_sample: usize);
}

/// Little-endian byte order
pub struct LittleEndian;

impl Endianness for LittleEndian {
    #[inline]
    fn i8_to_bytes(sample: i8) -> [u8; 1] {
        sample.to_le_bytes()
    }

    #[inline]
    fn i16_to_bytes(sample: i16) -> [u8; 2] {
        sample.to_le_bytes()
    }

    #[inline]
    fn i24_to_bytes(sample: i32) -> [u8; 3] {
        let unsigned: u32 = if sample >= 0 {
            sample as u32
        } else {
            0x800000 | ((sample - (-1 << 23)) as u32)
        };

        [
            (unsigned & 0xFF) as u8,
            ((unsigned & 0xFF00) >> 8) as u8,
            (unsigned >> 16) as u8,
        ]
    }

    #[inline]
    fn i32_to_bytes(sample: i32) -> [u8; 4] {
        sample.to_le_bytes()
    }

    #[inline]
    fn bytes_to_i8(bytes: [u8; 1]) -> i8 {
        i8::from_le_bytes(bytes)
    }

    #[inline]
    fn bytes_to_i16(bytes: [u8; 2]) -> i16 {
        i16::from_le_bytes(bytes)
    }

    #[inline]
    fn bytes_to_i24(bytes: [u8; 3]) -> i32 {
        let unsigned = ((bytes[2] as u32) << 16) | ((bytes[1] as u32) << 8) | bytes[0] as u32;

        if unsigned & 0x800000 == 0 {
            unsigned as i32
        } else {
            (unsigned & 0x7FFFFF) as i32 + (-1 << 23)
        }
    }

    #[inline]
    fn bytes_to_i32(bytes: [u8; 4]) -> i32 {
        i32::from_le_bytes(bytes)
    }

    fn bytes_to_be(buf: &mut [u8], bytes_per_sample: usize) {
        for chunk in buf.chunks_exact_mut(bytes_per_sample) {
            chunk.reverse();
        }
    }

    fn bytes_to_le(_buf: &mut [u8], _bytes_per_sample: usize) {
        // already little-endian, so nothing to do
    }
}

/// Big-endian byte order
pub struct BigEndian;

impl Endianness for BigEndian {
    #[inline]
    fn i8_to_bytes(sample: i8) -> [u8; 1] {
        sample.to_be_bytes()
    }

    #[inline]
    fn i16_to_bytes(sample: i16) -> [u8; 2] {
        sample.to_be_bytes()
    }

    #[inline]
    fn i24_to_bytes(sample: i32) -> [u8; 3] {
        let unsigned: u32 = if sample >= 0 {
            sample as u32
        } else {
            0x800000 | ((sample - (-1 << 23)) as u32)
        };

        [
            (unsigned >> 16) as u8,
            ((unsigned & 0xFF00) >> 8) as u8,
            (unsigned & 0xFF) as u8,
        ]
    }

    #[inline]
    fn i32_to_bytes(sample: i32) -> [u8; 4] {
        sample.to_be_bytes()
    }

    #[inline]
    fn bytes_to_i8(bytes: [u8; 1]) -> i8 {
        i8::from_be_bytes(bytes)
    }

    #[inline]
    fn bytes_to_i16(bytes: [u8; 2]) -> i16 {
        i16::from_be_bytes(bytes)
    }

    #[inline]
    fn bytes_to_i24(bytes: [u8; 3]) -> i32 {
        let unsigned = ((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | bytes[2] as u32;

        if unsigned & 0x800000 == 0 {
            unsigned as i32
        } else {
            (unsigned & 0x7FFFFF) as i32 + (-1 << 23)
        }
    }

    #[inline]
    fn bytes_to_i32(bytes: [u8; 4]) -> i32 {
        i32::from_be_bytes(bytes)
    }

    fn bytes_to_be(_buf: &mut [u8], _bytes_per_sample: usize) {
        // already big-endian, so nothing to do
    }

    fn bytes_to_le(buf: &mut [u8], bytes_per_sample: usize) {
        for chunk in buf.chunks_exact_mut(bytes_per_sample) {
            chunk.reverse();
        }
    }
}

#[allow(unused)]
fn test_endianness<F: bitstream_io::Endianness, E: Endianness>() {
    use bitstream_io::{BitWrite, BitWriter};

    // 8 bits-per-sample to bytes
    for i in i8::MIN..=i8::MAX {
        let mut buf1 = [0; 1];
        let mut w: BitWriter<_, F> = BitWriter::new(buf1.as_mut_slice());
        w.write::<8, i8>(i).unwrap();

        let buf2 = E::i8_to_bytes(i);

        assert_eq!(buf1, buf2);

        let j = E::bytes_to_i8(buf2);
        assert_eq!(i, j);
    }

    // 16 bits-per-sample to bytes
    for i in i16::MIN..=i16::MAX {
        let mut buf1 = [0; 2];
        let mut w: BitWriter<_, F> = BitWriter::new(buf1.as_mut_slice());
        w.write::<16, i16>(i).unwrap();

        let buf2 = E::i16_to_bytes(i);

        assert_eq!(buf1, buf2);

        let j = E::bytes_to_i16(buf2);
        assert_eq!(i, j);
    }

    // 24 bits-per-sample to bytes
    for i in (-1 << 23)..=((1 << 23) - 1) {
        let mut buf1 = [0; 3];
        let mut w: BitWriter<_, F> = BitWriter::new(buf1.as_mut_slice());
        w.write::<24, i32>(i).unwrap();

        let buf2 = E::i24_to_bytes(i);

        assert_eq!(buf1, buf2);

        let j = E::bytes_to_i24(buf2);
        assert_eq!(i, j);
    }
}

#[test]
fn test_samples_le() {
    test_endianness::<bitstream_io::LittleEndian, LittleEndian>()
}

#[test]
fn test_samples_be() {
    test_endianness::<bitstream_io::BigEndian, BigEndian>()
}

