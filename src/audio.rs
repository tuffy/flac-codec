// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Storage for PCM samples

use crate::metadata::Streaminfo;

/// A decoded set of audio samples
#[derive(Clone, Default, Debug)]
pub struct Frame {
    /// One outer Vec per channel. Each channel Vec stores samples.
    pub channels: Vec<Vec<i32>>,

    /// Sample rate in Hz
    pub sample_rate: u32,

    /// Number of bits-per-sample, from 4 to 32
    pub bits_per_sample: u8,
}

impl Frame {
    /// Empties frame of its contents and returns it
    pub fn empty(mut self, streaminfo: &Streaminfo) -> Self {
        self.channels
            .resize_with(streaminfo.channels.get().into(), || vec![]);
        self.channels.iter_mut().for_each(|c| c.resize(0, 0));
        self.sample_rate = streaminfo.sample_rate;
        self.bits_per_sample = u32::from(streaminfo.bits_per_sample) as u8;
        self
    }

    /// Returns true if frame is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self.channels.get(0) {
            Some(c) => c.is_empty(),
            None => true,
        }
    }
}
