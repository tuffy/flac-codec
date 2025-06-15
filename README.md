flac-codec
==========

An comprehensive Rust library for handling FLAC files.

This library implements [RFC9639](https://www.ietf.org/rfc/rfc9639.html)
to process FLAC files according to the standard in a safe, performant
fashion with a straightforward API.

## Handles Metadata Blocks

- Read any single metadata block from a file
- Read all the metadata blocks from a file
- Update a file's metadata blocks with a single function call

## Decodes Files

- Decode FLAC files to bytes in some endianness
  (useful for storing in other PCM containers, like .wav files)
- Decode FLAC files to signed integer samples
  (audio playback system libraries often require these)
- Decode subset FLAC files from raw "subset" streams
- Can decode from paths on disk, or from any input stream that
  implements `std::io::Read`
- Provides an easy reading API
  - If you need bytes, the byte reader simply implements `std::io::Read`
  - If you need samples, a `Read`-like trait for signed integers is also provided
- Offers seekable reader variants if the underlying stream is also seekable

## Encodes Files

- Encode FLAC files from bytes in some endianness
  (again, useful for encoding data from raw PCM containers)
- Encode FLAC files from signed integer samples
- Encode FLAC files into raw "subset" streams
- Can encode to paths on disk, or to any output stream
  that implements both `std::io::Write` and `std::io::Seek`
- Provides an easy writing API
  - If you can provide bytes, there's an encoder that implements `std::io::Write`
  - If you can provide samples, a `Write`-like interface is also provided
- Encoding process modeled on the reference implementation's
  and achieves similar compression when using identical parameters
- Offers multithreaded encoding via the optional `rayon` feature for better performance

## Analyzes Files

- Parses FLAC files to Rust data structures
- Can rebuild FLAC files from those same data structures
- Is able to round-trip files that are byte-for-byte identical to the originals

## Tested

- Verified against tests ported from the reference implementation
- Handles the entire [FLAC decoder testbench suite](https://github.com/ietf-wg-cellar/flac-test-files)
- Tested against my personal collection of nearly 200,000 FLAC files
