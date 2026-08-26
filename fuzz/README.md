# Codec fuzzing

Run the raw decoder target with the crate fixtures and external decoder corpus as seeds:

```console
cargo +nightly fuzz run decode_raw tests/data flac-test-files/subset flac-test-files/uncommon flac-test-files/faulty -- -max_len=16777216
```

Run the structure-aware target to preserve valid checksums while mutating stream invariants:

```console
cargo +nightly fuzz run decode_structured
```

Exercise seek-table and post-seek decoding paths with the decoder corpus:

```console
cargo +nightly fuzz run decode_seek tests/data flac-test-files/subset flac-test-files/uncommon
```

Exercise encoder options, all bit depths, channel counts, and short-block boundaries while
asserting exact decoder round trips:

```console
cargo +nightly fuzz run encode_roundtrip
```
