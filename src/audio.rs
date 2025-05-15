// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Storage for PCM samples

use std::ops::Index;

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

/// A decoded set of audio samples
#[derive(Clone, Default, Debug, Eq, PartialEq)]
pub struct Frame {
    // all samples, stacked by channel
    samples: Vec<i32>,

    // total number of channels
    channels: usize,

    // total length of each channel in samples
    channel_len: usize,

    // bits-per-sample
    bits_per_sample: u32,
}

impl Frame {
    /// Returns channel count
    #[inline]
    pub fn channel_count(&self) -> usize {
        self.channels
    }

    /// Returns number of bits per sample
    #[inline]
    pub fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }

    /// Returns PCM frame count
    #[inline]
    pub fn pcm_frames(&self) -> usize {
        self.channel_len
    }

    /// Returns empty Frame which can be filled as needed
    #[inline]
    pub fn empty(channels: usize, bits_per_sample: u32) -> Self {
        Self {
            samples: Vec::new(),
            channels,
            channel_len: 0,
            bits_per_sample,
        }
    }

    fn resize(&mut self, bits_per_sample: u32, channels: usize, block_size: usize) -> &mut [i32] {
        self.bits_per_sample = bits_per_sample;
        self.channels = channels;
        self.channel_len = block_size;
        self.samples.resize(channels * block_size, 0);
        &mut self.samples
    }

    /// Resizes our frame with the given parameters and returns channel iterator
    pub fn resized_channels(
        &mut self,
        bits_per_sample: u32,
        channels: usize,
        block_size: usize,
    ) -> impl Iterator<Item = &mut [i32]> {
        self.resize(bits_per_sample, channels, block_size)
            .chunks_exact_mut(block_size)
    }

    /// Resizes our frame for two channels and returns both
    pub fn resized_stereo(
        &mut self,
        bits_per_sample: u32,
        block_size: usize,
    ) -> (&mut [i32], &mut [i32]) {
        self.resize(bits_per_sample, 2, block_size)
            .split_at_mut(block_size)
    }

    /// Iterates over any samples in interleaved order
    pub fn iter(&self) -> impl Iterator<Item = i32> {
        (0..self.samples.len()).map(|i| {
            let (sample, channel) = (i / self.channels, i % self.channels);
            self.samples[channel * self.channel_len + sample]
        })
    }

    /// Iterates over all channels
    pub fn channels(&self) -> impl Iterator<Item = &[i32]> {
        self.samples.chunks_exact(self.channel_len)
    }

    /// Returns bytes-per-sample
    pub fn bytes_per_sample(&self) -> usize {
        self.bits_per_sample().div_ceil(8) as usize
    }

    /// Returns total length of buffer in bytes
    #[inline]
    pub fn bytes_len(&self) -> usize {
        self.bytes_per_sample() * self.samples.len()
    }

    /// Fills buffer with our samples in the given endianness
    pub fn to_buf<E: Endianness>(&self, buf: &mut [u8]) {
        // TODO - replace these with array_chunks_mut once that stabilizes

        match self.bytes_per_sample() {
            1 => {
                for (sample, bytes) in self.iter().zip(buf.chunks_exact_mut(1)) {
                    bytes.copy_from_slice(&E::i8_to_bytes(sample as i8));
                }
            }
            2 => {
                for (sample, bytes) in self.iter().zip(buf.chunks_exact_mut(2)) {
                    bytes.copy_from_slice(&E::i16_to_bytes(sample as i16));
                }
            }
            3 => {
                for (sample, bytes) in self.iter().zip(buf.chunks_exact_mut(3)) {
                    bytes.copy_from_slice(&E::i24_to_bytes(sample));
                }
            }
            4 => {
                for (sample, bytes) in self.iter().zip(buf.chunks_exact_mut(4)) {
                    bytes.copy_from_slice(&E::i32_to_bytes(sample));
                }
            }
            _ => panic!("unsupported number of bytes per sample"),
        }
    }

    /// Fills frame samples from bytes of the given endianness
    pub fn from_buf<E: Endianness>(&mut self, buf: &[u8]) -> &Self {
        fn buf_chunks<const BYTES_PER_SAMPLE: usize>(
            channels: usize,
            channel_len: usize,
            buf: &[u8],
        ) -> impl Iterator<Item = [u8; BYTES_PER_SAMPLE]> {
            (0..channels).flat_map(move |c| {
                (0..channel_len).map(move |s| {
                    buf[((s * channels) + c) * BYTES_PER_SAMPLE
                        ..((s * channels) + c + 1) * BYTES_PER_SAMPLE]
                        .try_into()
                        .unwrap()
                })
            })
        }

        match self.bytes_per_sample() {
            1 => {
                self.channel_len = buf.len() / self.channels;
                self.samples.resize(buf.len(), 0);

                for (sample, bytes) in self.samples.iter_mut().zip(buf_chunks::<1>(
                    self.channels,
                    self.channel_len,
                    buf,
                )) {
                    *sample = E::bytes_to_i8(bytes) as i32
                }
            }
            2 => {
                self.channel_len = (buf.len() / 2) / self.channels;
                self.samples.resize(buf.len() / 2, 0);

                for (sample, bytes) in self.samples.iter_mut().zip(buf_chunks::<2>(
                    self.channels,
                    self.channel_len,
                    buf,
                )) {
                    *sample = E::bytes_to_i16(bytes) as i32
                }
            }
            3 => {
                self.channel_len = (buf.len() / 3) / self.channels;
                self.samples.resize(buf.len() / 3, 0);

                for (sample, bytes) in self.samples.iter_mut().zip(buf_chunks::<3>(
                    self.channels,
                    self.channel_len,
                    buf,
                )) {
                    *sample = E::bytes_to_i24(bytes)
                }
            }
            4 => {
                self.channel_len = (buf.len() / 4) / self.channels;
                self.samples.resize(buf.len() / 4, 0);

                for (sample, bytes) in self.samples.iter_mut().zip(buf_chunks::<4>(
                    self.channels,
                    self.channel_len,
                    buf,
                )) {
                    *sample = E::bytes_to_i32(bytes)
                }
            }
            _ => panic!("unsupported number of bytes per sample"),
        }

        self
    }
}

#[allow(unused)]
fn test_buffer<E: Endianness>() {
    fn test_buf<const BYTES_PER_SAMPLE: usize, E: Endianness>(channels: usize, samples: usize) {
        let frame1 = Frame {
            samples: (0..samples).map(|i| i as i32).collect(),
            channels,
            channel_len: samples / channels,
            bits_per_sample: (BYTES_PER_SAMPLE * 8) as u32,
        };

        let mut buf = vec![0; frame1.bytes_len()];

        frame1.to_buf::<E>(&mut buf);

        let mut frame2 = Frame::empty(channels, (BYTES_PER_SAMPLE * 8) as u32);
        frame2.from_buf::<E>(&buf);

        assert_eq!(frame1, frame2);
    }

    test_buf::<1, E>(1, 50);
    test_buf::<2, E>(1, 50);
    test_buf::<3, E>(1, 50);
    test_buf::<4, E>(1, 50);

    test_buf::<1, E>(2, 50);
    test_buf::<2, E>(2, 50);
    test_buf::<3, E>(2, 50);
    test_buf::<4, E>(2, 50);

    test_buf::<1, E>(3, 60);
    test_buf::<2, E>(3, 60);
    test_buf::<3, E>(3, 60);
    test_buf::<4, E>(3, 60);

    test_buf::<1, E>(4, 80);
    test_buf::<2, E>(4, 80);
    test_buf::<3, E>(4, 80);
    test_buf::<4, E>(4, 80);

    test_buf::<1, E>(5, 80);
    test_buf::<2, E>(5, 80);
    test_buf::<3, E>(5, 80);
    test_buf::<4, E>(5, 80);

    test_buf::<1, E>(6, 60);
    test_buf::<2, E>(6, 60);
    test_buf::<3, E>(6, 60);
    test_buf::<4, E>(6, 60);

    test_buf::<1, E>(7, 70);
    test_buf::<2, E>(7, 70);
    test_buf::<3, E>(7, 70);
    test_buf::<4, E>(7, 70);

    test_buf::<1, E>(8, 80);
    test_buf::<2, E>(8, 80);
    test_buf::<3, E>(8, 80);
    test_buf::<4, E>(8, 80);
}

#[test]
fn test_buffer_le() {
    test_buffer::<LittleEndian>()
}

#[test]
fn test_buffer_be() {
    test_buffer::<BigEndian>()
}

/// Returns given channel's samples
impl Index<usize> for Frame {
    type Output = [i32];

    fn index(&self, index: usize) -> &[i32] {
        &self.samples[index * self.channel_len..(index + 1) * self.channel_len]
    }
}
