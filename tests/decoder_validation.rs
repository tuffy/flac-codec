use flac_codec::{
    byteorder::LittleEndian,
    decode::{FlacByteReader, FlacSampleReader, Verified, verify_reader},
    encode::{FlacSampleWriter, Options},
    metadata::{BlockList, Streaminfo, write_blocks},
    stream::{ChannelAssignment, Frame, FrameIterator, FrameNumber, Subframe, SubframeWidth},
};
use std::io::{Cursor, sink};
use std::num::NonZero;

fn mutate_streaminfo(mutator: impl FnOnce(&mut Streaminfo)) -> Vec<u8> {
    let source = include_bytes!("data/sine.flac").as_slice();
    let mut input = Cursor::new(source);
    let mut blocks = BlockList::read(&mut input).unwrap();
    let frames_start = usize::try_from(input.position()).unwrap();
    mutator(blocks.streaminfo_mut());

    let mut output = Vec::new();
    write_blocks(&mut output, blocks).unwrap();
    output.extend_from_slice(&source[frames_start..]);
    output
}

fn mutate_source_frames(
    source: &[u8],
    mutator: impl FnOnce(&mut BlockList, &mut [Frame]),
) -> Vec<u8> {
    let mut blocks = BlockList::read(source).unwrap();
    let streaminfo = blocks.streaminfo().clone();
    let mut frames = FrameIterator::new(source)
        .unwrap()
        .map(|frame| frame.map(|(frame, _)| frame))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    mutator(&mut blocks, &mut frames);

    let mut output = Vec::new();
    write_blocks(&mut output, blocks).unwrap();
    for frame in frames {
        frame.write(&streaminfo, &mut output).unwrap();
    }
    output
}

fn mutate_frames(mutator: impl FnOnce(&mut BlockList, &mut [Frame])) -> Vec<u8> {
    mutate_source_frames(include_bytes!("data/sine.flac"), mutator)
}

fn assert_decoders_reject(flac: &[u8]) {
    assert!(verify_reader(flac).is_err());

    let byte_result = FlacByteReader::endian(flac, LittleEndian)
        .and_then(|mut reader| std::io::copy(&mut reader, &mut sink()).map_err(Into::into));
    assert!(byte_result.is_err());

    let sample_result = FlacSampleReader::new(flac).and_then(|mut reader| {
        let mut samples = Vec::new();
        reader.read_to_end(&mut samples).map(|_| ())
    });
    assert!(sample_result.is_err());

    let frame_result = FrameIterator::new(flac)
        .and_then(|frames| frames.collect::<Result<Vec<_>, _>>().map(|_| ()));
    assert!(frame_result.is_err());
}

#[test]
fn test_streaminfo_maximum_frame_size() {
    let flac = mutate_streaminfo(|streaminfo| {
        streaminfo.maximum_frame_size = NonZero::new(1);
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_streaminfo_minimum_frame_size() {
    let flac = mutate_streaminfo(|streaminfo| {
        streaminfo.minimum_frame_size = Some(NonZero::new(Streaminfo::MAX_FRAME_SIZE).unwrap());
        streaminfo.maximum_frame_size = Some(NonZero::new(Streaminfo::MAX_FRAME_SIZE).unwrap());
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_streaminfo_minimum_block_size() {
    let flac = mutate_streaminfo(|streaminfo| {
        streaminfo.minimum_block_size = u16::MAX;
        streaminfo.maximum_block_size = u16::MAX;
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_streaminfo_total_samples_smaller_than_frame() {
    let flac = mutate_streaminfo(|streaminfo| {
        streaminfo.total_samples = NonZero::new(1);
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_short_block_must_be_last() {
    let mut source = Cursor::new(Vec::new());
    let mut writer = FlacSampleWriter::new(
        &mut source,
        Options::fast().block_size(16).unwrap().no_padding(),
        44100,
        16,
        1,
        Some(33),
    )
    .unwrap();
    writer.write(&(0..33).collect::<Vec<_>>()).unwrap();
    writer.finalize().unwrap();

    let flac = mutate_source_frames(source.get_ref(), |blocks, frames| {
        blocks.streaminfo_mut().minimum_frame_size = None;
        blocks.streaminfo_mut().maximum_frame_size = None;
        frames.rotate_right(1);
        for (frame_number, frame) in frames.iter_mut().enumerate() {
            frame.header.blocking_strategy = false;
            frame.header.frame_number = FrameNumber(frame_number as u64);
        }
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_fixed_block_frame_number_sequence() {
    let flac = mutate_frames(|blocks, frames| {
        blocks.streaminfo_mut().minimum_frame_size = None;
        blocks.streaminfo_mut().maximum_frame_size = None;
        frames[1].header.frame_number = frames[0].header.frame_number;
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_variable_block_sample_number_sequence() {
    let flac = mutate_frames(|blocks, frames| {
        blocks.streaminfo_mut().minimum_frame_size = None;
        blocks.streaminfo_mut().maximum_frame_size = None;

        let mut sample_offset = 0;
        for frame in frames.iter_mut() {
            frame.header.blocking_strategy = true;
            frame.header.frame_number = FrameNumber(sample_offset);
            sample_offset += u64::from(u16::from(frame.header.block_size));
        }
        frames[1].header.frame_number.0 += 1;
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_blocking_strategy_change() {
    let flac = mutate_frames(|blocks, frames| {
        blocks.streaminfo_mut().minimum_frame_size = None;
        blocks.streaminfo_mut().maximum_frame_size = None;
        frames[1].header.blocking_strategy = true;
        frames[1].header.frame_number =
            FrameNumber(u64::from(u16::from(frames[0].header.block_size)));
    });
    assert_decoders_reject(&flac);
}

#[test]
fn test_variable_block_sample_number_control() {
    let flac = mutate_frames(|blocks, frames| {
        blocks.streaminfo_mut().minimum_frame_size = None;
        blocks.streaminfo_mut().maximum_frame_size = None;

        let mut sample_offset = 0;
        for frame in frames {
            frame.header.blocking_strategy = true;
            frame.header.frame_number = FrameNumber(sample_offset);
            sample_offset += u64::from(u16::from(frame.header.block_size));
        }
    });
    assert_eq!(verify_reader(flac.as_slice()).unwrap(), Verified::MD5Match);
}

fn invalid_mid_side_stream(bits_per_sample: u32, side: SubframeWidth) -> Vec<u8> {
    let mut source = Cursor::new(Vec::new());
    let mut writer = FlacSampleWriter::new(
        &mut source,
        Options::fast().block_size(16).unwrap().no_padding(),
        44_100,
        bits_per_sample,
        2,
        Some(32),
    )
    .unwrap();
    writer.write(&[0; 32]).unwrap();
    writer.finalize().unwrap();

    mutate_source_frames(source.get_ref(), |blocks, frames| {
        blocks.streaminfo_mut().minimum_frame_size = None;
        blocks.streaminfo_mut().maximum_frame_size = None;
        let block_size = u16::from(frames[0].header.block_size);
        frames[0].header.channel_assignment = ChannelAssignment::MidSide;
        frames[0].subframes = vec![
            SubframeWidth::Common(Subframe::Constant {
                block_size,
                sample: 0,
                wasted_bps: 0,
            }),
            side,
        ];
    })
}

#[test]
fn reject_31_bit_mid_side_reconstruction_overflow() {
    let flac = invalid_mid_side_stream(
        31,
        SubframeWidth::Common(Subframe::Constant {
            block_size: 16,
            sample: i32::MIN,
            wasted_bps: 0,
        }),
    );

    assert!(verify_reader(flac.as_slice()).is_err());
}

#[test]
fn reject_32_bit_mid_side_reconstruction_overflow() {
    let flac = invalid_mid_side_stream(
        32,
        SubframeWidth::Wide(Subframe::Constant {
            block_size: 16,
            sample: (1i64 << 32) - 1,
            wasted_bps: 0,
        }),
    );

    assert!(verify_reader(flac.as_slice()).is_err());
}
