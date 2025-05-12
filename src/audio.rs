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
}

/// Little-endian byte order
pub struct LittleEndian;

impl Endianness for LittleEndian {
    fn i8_to_bytes(sample: i8) -> [u8; 1] {
        sample.to_le_bytes()
    }

    fn i16_to_bytes(sample: i16) -> [u8; 2] {
        sample.to_le_bytes()
    }

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

    fn i32_to_bytes(sample: i32) -> [u8; 4] {
        sample.to_le_bytes()
    }
}

#[allow(unused)]
fn test_endianness<F: bitstream_io::Endianness, E: Endianness>() {
    use bitstream_io::{BitWrite, BitWriter};

    // 8 bits-per-sample
    for i in i8::MIN..=i8::MAX {
        let mut buf1 = [0; 1];
        let mut w: BitWriter<_, F> = BitWriter::new(buf1.as_mut_slice());
        w.write::<8, i8>(i).unwrap();

        let buf2 = E::i8_to_bytes(i);

        assert_eq!(buf1, buf2);
    }

    // 16 bits-per-sample
    for i in i16::MIN..=i16::MAX {
        let mut buf1 = [0; 2];
        let mut w: BitWriter<_, F> = BitWriter::new(buf1.as_mut_slice());
        w.write::<16, i16>(i).unwrap();

        let buf2 = E::i16_to_bytes(i);

        assert_eq!(buf1, buf2);
    }

    // 24 bits-per-sample
    for i in (-1 << 23)..=((1 << 23) - 1) {
        let mut buf1 = [0; 3];
        let mut w: BitWriter<_, F> = BitWriter::new(buf1.as_mut_slice());
        w.write::<24, i32>(i).unwrap();

        let buf2 = E::i24_to_bytes(i);

        assert_eq!(buf1, buf2);
    }
}

#[test]
fn test_samples_le() {
    test_endianness::<bitstream_io::LittleEndian, LittleEndian>()
}

/// A decoded set of audio samples
#[derive(Clone, Default, Debug)]
pub struct Frame {
    // all samples, stacked by channel
    samples: Vec<i32>,

    // total number of channels
    channels: usize,

    // total length of each channel in samples
    channel_len: usize,

    // bits-per-sample
    bits_per_sample: u32,

    // sample rate, in Hz
    sample_rate: u32,
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

    /// Returns sample rate in Hz
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns PCM frame count
    #[inline]
    pub fn pcm_frames(&self) -> usize {
        self.channel_len
    }

    /// Empties frame of its contents and returns it
    #[inline]
    pub fn empty(mut self) -> Self {
        self.samples.clear();
        self.channels = 0;
        self.channel_len = 0;
        self.sample_rate = 0;
        self.bits_per_sample = 0;
        self
    }

    /// Returns true if frame is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn resize(
        &mut self,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: usize,
        block_size: usize,
    ) -> &mut [i32] {
        self.sample_rate = sample_rate;
        self.bits_per_sample = bits_per_sample;
        self.channels = channels;
        self.channel_len = block_size;
        self.samples.resize(channels * block_size, 0);
        &mut self.samples
    }

    /// Resizes our frame with the given parameters and returns channel iterator
    pub fn resized_channels(
        &mut self,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: usize,
        block_size: usize,
    ) -> impl Iterator<Item = &mut [i32]> {
        self.resize(sample_rate, bits_per_sample, channels, block_size)
            .chunks_exact_mut(block_size)
    }

    /// Resizes our frame for two channels and returns both
    pub fn resized_stereo(
        &mut self,
        sample_rate: u32,
        bits_per_sample: u32,
        block_size: usize,
    ) -> (&mut [i32], &mut [i32]) {
        self.resize(sample_rate, bits_per_sample, 2, block_size)
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
    pub fn fill_buf<E: Endianness>(&self, buf: &mut [u8]) {
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
            _ => panic!("unsupported number of bytes per samples"),
        }
    }
}

/// Returns given channel's samples
impl Index<usize> for Frame {
    type Output = [i32];

    fn index(&self, index: usize) -> &[i32] {
        &self.samples[index * self.channel_len..(index + 1) * self.channel_len]
    }
}
