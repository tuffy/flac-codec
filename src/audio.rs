// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Storage for PCM samples

use std::ops::Index;

/// A decoded set of audio samples
#[derive(Clone, Default, Debug)]
pub struct Frame {
    // all samples, stacked by channel
    samples: Vec<i32>,

    // total number of channels
    channels: usize,

    // total length of each channel in samples
    channel_len: usize,

    // channel_len = self.samples.len() / self.channels;

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
        // self.samples.len().checked_div(self.channels).unwrap_or_default()
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

    /// Resizes our frame with the given parameters and returns channel iteraetor
    pub fn resize_for(
        &mut self,
        sample_rate: u32,
        bits_per_sample: u32,
        channels: usize,
        block_size: usize,
    ) -> impl Iterator<Item = &mut [i32]> {
        self.sample_rate = sample_rate;
        self.bits_per_sample = bits_per_sample;
        self.channels = channels;
        self.channel_len = block_size;
        self.samples.resize(channels * block_size, 0);
        self.samples.chunks_exact_mut(block_size)
    }

    /// Resizes our frame with the given parameters and returns channel iteraetor
    pub fn resize_for_2(
        &mut self,
        sample_rate: u32,
        bits_per_sample: u32,
        block_size: usize,
    ) -> (&mut [i32], &mut [i32]) {
        self.sample_rate = sample_rate;
        self.bits_per_sample = bits_per_sample;
        self.channels = 2;
        self.channel_len = block_size;
        self.samples.resize(2 * block_size, 0);
        self.samples.split_at_mut(block_size)
    }

    /// Iterates over any samples in interleaved order
    pub fn iter(&self) -> impl Iterator<Item = i32> {
        (0..self.samples.len()).map(|i| {
            let (sample, channel) = (i / self.channels, i % self.channels);
            self.samples[channel * self.channel_len + sample]
        })
    }
}

/// Returns given channel's samples
impl Index<usize> for Frame {
    type Output = [i32];

    fn index(&self, index: usize) -> &[i32] {
        &self.samples[index * self.channel_len..(index + 1) * self.channel_len]
    }
}
