// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For handling PCM audio in frame-sized chunks

use arrayvec::ArrayVec;

#[derive(Clone, Default)]
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

    /// Returns PCM frame count
    #[inline]
    pub fn pcm_frames(&self) -> usize {
        self.channel_len
    }

    pub fn resize(
        &mut self,
        bits_per_sample: u32,
        channels: usize,
        block_size: usize,
    ) -> &mut [i32] {
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

    /// Returns bytes-per-sample
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        self.bits_per_sample.div_ceil(8) as usize
    }

    /// Returns total length of buffer in bytes
    #[inline]
    pub fn bytes_len(&self) -> usize {
        self.bytes_per_sample() * self.samples.len()
    }

    /// Iterates over any samples in interleaved order
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = i32> {
        MultiZip {
            iters: self.channels().map(|c| c.iter().copied()).collect(),
        }
        .flatten()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut i32> {
        MultiZip {
            iters: self.channels_mut().map(|c| c.iter_mut()).collect(),
        }
        .flatten()
    }

    /// Fills buffer with our samples in the given endianness
    pub fn to_buf<E: crate::byteorder::Endianness>(&self, buf: &mut [u8]) {
        match self.bytes_per_sample() {
            1 => {
                for (sample, bytes) in self.iter().zip(buf.as_chunks_mut().0) {
                    *bytes = E::i8_to_bytes(sample as i8);
                }
            }
            2 => {
                for (sample, bytes) in self.iter().zip(buf.as_chunks_mut().0) {
                    *bytes = E::i16_to_bytes(sample as i16);
                }
            }
            3 => {
                for (sample, bytes) in self.iter().zip(buf.as_chunks_mut().0) {
                    *bytes = E::i24_to_bytes(sample);
                }
            }
            4 => {
                for (sample, bytes) in self.iter().zip(buf.as_chunks_mut().0) {
                    *bytes = E::i32_to_bytes(sample);
                }
            }
            _ => panic!("unsupported number of bytes per sample"),
        }
    }

    /// Iterates over all channels
    #[inline]
    pub fn channels(&self) -> impl Iterator<Item = &[i32]> {
        self.samples.chunks_exact(self.channel_len)
    }

    /// Iterates over all channels as mutable samples
    #[inline]
    pub fn channels_mut(&mut self) -> impl Iterator<Item = &mut [i32]> {
        self.samples.chunks_exact_mut(self.channel_len)
    }

    /// Fills frame samples from bytes of the given endianness
    pub fn fill_from_buf<E: crate::byteorder::Endianness>(&mut self, buf: &[u8]) -> &Self {
        match self.bytes_per_sample() {
            1 => {
                self.channel_len = buf.len() / self.channels;
                self.samples.resize(buf.len(), 0);

                for (sample, bytes) in self.iter_mut().zip(buf.as_chunks().0) {
                    *sample = E::bytes_to_i8(*bytes).into()
                }
            }
            2 => {
                self.channel_len = (buf.len() / 2) / self.channels;
                self.samples.resize(buf.len() / 2, 0);

                for (sample, bytes) in self.iter_mut().zip(buf.as_chunks().0) {
                    *sample = E::bytes_to_i16(*bytes).into()
                }
            }
            3 => {
                self.channel_len = (buf.len() / 3) / self.channels;
                self.samples.resize(buf.len() / 3, 0);

                for (sample, bytes) in self.iter_mut().zip(buf.as_chunks().0) {
                    *sample = E::bytes_to_i24(*bytes).into()
                }
            }
            4 => {
                self.channel_len = (buf.len() / 4) / self.channels;
                self.samples.resize(buf.len() / 4, 0);

                for (sample, bytes) in self.iter_mut().zip(buf.as_chunks().0) {
                    *sample = E::bytes_to_i32(*bytes).into()
                }
            }
            _ => panic!("unsupported number of bytes per sample"),
        }

        self
    }

    /// Fills frame samples from interleaved samples
    pub fn fill_from_samples(&mut self, samples: &[i32]) -> &Self {
        self.channel_len = samples.len() / self.channels;
        self.samples.resize(samples.len(), 0);

        for (o, i) in self.iter_mut().zip(samples) {
            *o = *i;
        }

        self
    }
}

const MAX_CHANNELS: usize = 8;

struct MultiZip<I> {
    iters: ArrayVec<I, MAX_CHANNELS>,
}

impl<I: Iterator> Iterator for MultiZip<I> {
    type Item = ArrayVec<I::Item, MAX_CHANNELS>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iters.iter_mut().map(|i| i.next()).collect()
    }
}
