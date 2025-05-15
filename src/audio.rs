#[derive(Default)]
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

    pub fn resize(&mut self, bits_per_sample: u32, channels: usize, block_size: usize) -> &mut [i32] {
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
    pub fn iter(&self) -> impl Iterator<Item = i32> {
        // TODO - use MultiZip
        (0..self.samples.len()).map(|i| {
            let (sample, channel) = (i / self.channels, i % self.channels);
            self.samples[channel * self.channel_len + sample]
        })
    }

    /// Fills buffer with our samples in the given endianness
    pub fn to_buf<E: crate::byteorder::Endianness>(&self, buf: &mut [u8]) {
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

    /// Iterates over all channels
    #[inline]
    pub fn channels(&self) -> impl Iterator<Item = &[i32]> {
        self.samples.chunks_exact(self.channel_len)
    }

    /// Fills frame samples from bytes of the given endianness
    pub fn fill_from_buf<E: crate::byteorder::Endianness>(&mut self, buf: &[u8]) -> &Self {
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

    /// Fills frame samples from interleaved samples
    pub fn fill_from_samples(&mut self, samples: &[i32]) -> &Self {
        fn samples_iter(
            channels: usize,
            channel_len: usize,
            samples: &[i32],
        ) -> impl Iterator<Item = i32> {
            (0..channels)
                .flat_map(move |c| (0..channel_len).map(move |s| samples[(s * channels) + c]))
        }

        self.channel_len = samples.len() / self.channels;
        self.samples.resize(samples.len(), 0);

        for (o, i) in
            self.samples
                .iter_mut()
                .zip(samples_iter(self.channels, self.channel_len, samples))
        {
            *o = i;
        }

        self
    }
}

