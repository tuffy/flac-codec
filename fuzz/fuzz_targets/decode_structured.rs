#![no_main]

use flac_codec::{
    byteorder::LittleEndian,
    decode::{FlacByteReader, FlacSampleReader, verify_reader},
    encode::{FlacSampleWriter, Options},
    metadata::{BlockList, Streaminfo, write_blocks},
    stream::{FrameIterator, FrameNumber},
};
use libfuzzer_sys::fuzz_target;
use std::io::{Cursor, sink};
use std::num::NonZero;
use std::sync::OnceLock;

fn base_stream() -> &'static [u8] {
    static BASE: OnceLock<Vec<u8>> = OnceLock::new();

    BASE.get_or_init(|| {
        let mut output = Cursor::new(Vec::new());
        let mut writer = FlacSampleWriter::new(
            &mut output,
            Options::fast().block_size(16).unwrap().no_padding(),
            44100,
            16,
            1,
            Some(33),
        )
        .unwrap();
        writer.write(&(0..33).collect::<Vec<_>>()).unwrap();
        writer.finalize().unwrap();
        output.into_inner()
    })
}

fn mutated_stream(selector: u8) -> Vec<u8> {
    let source = base_stream();
    let mut blocks = BlockList::read(source).unwrap();
    let mut frames = FrameIterator::new(source)
        .unwrap()
        .map(|frame| frame.map(|(frame, _)| frame))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    match selector % 7 {
        0 => blocks.streaminfo_mut().maximum_frame_size = NonZero::new(1),
        1 => {
            blocks.streaminfo_mut().minimum_frame_size =
                Some(NonZero::new(Streaminfo::MAX_FRAME_SIZE).unwrap());
            blocks.streaminfo_mut().maximum_frame_size =
                Some(NonZero::new(Streaminfo::MAX_FRAME_SIZE).unwrap());
        }
        2 => {
            blocks.streaminfo_mut().minimum_block_size = u16::MAX;
            blocks.streaminfo_mut().maximum_block_size = u16::MAX;
        }
        3 => blocks.streaminfo_mut().total_samples = NonZero::new(1),
        4 => {
            blocks.streaminfo_mut().minimum_frame_size = None;
            blocks.streaminfo_mut().maximum_frame_size = None;
            frames[1].header.frame_number = frames[0].header.frame_number;
        }
        5 => {
            blocks.streaminfo_mut().minimum_frame_size = None;
            blocks.streaminfo_mut().maximum_frame_size = None;
            let mut sample_offset = 0;
            for frame in &mut frames {
                frame.header.blocking_strategy = true;
                frame.header.frame_number = FrameNumber(sample_offset);
                sample_offset += u64::from(u16::from(frame.header.block_size));
            }
            frames[1].header.frame_number.0 += 1;
        }
        6 => {
            blocks.streaminfo_mut().minimum_frame_size = None;
            blocks.streaminfo_mut().maximum_frame_size = None;
            frames[1].header.blocking_strategy = true;
            frames[1].header.frame_number =
                FrameNumber(u64::from(u16::from(frames[0].header.block_size)));
        }
        _ => unreachable!(),
    }

    let streaminfo = blocks.streaminfo().clone();
    let mut output = Vec::new();
    write_blocks(&mut output, blocks).unwrap();
    for frame in frames {
        frame.write(&streaminfo, &mut output).unwrap();
    }
    output
}

fn assert_rejected(flac: &[u8]) {
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

fuzz_target!(|data: &[u8]| {
    let selector = data.first().copied().unwrap_or_default();
    assert_rejected(&mutated_stream(selector));
});
