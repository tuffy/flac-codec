use flac_codec::{
    Error,
    metadata::{Application, Cuesheet, Padding, Picture, SeekTable, Streaminfo, VorbisComment},
};
use std::path::Path;

/// This corresponds to the reference implementation's
///
/// "metaflac --list <file1.flac> <file2.flac> ..."
///
/// Except that I display the input path first
/// instead of before each line, which is
/// a little easier to follow in this simple example.

fn main() {
    for flac in std::env::args_os().skip(1) {
        if let Err(err) = display_blocks(&flac) {
            eprintln!("* {}: {err}", flac.display());
        }
    }
}

fn display_blocks<P: AsRef<Path>>(flac: P) -> Result<(), Error> {
    use flac_codec::metadata::{Block, blocks};

    println!("{}:", flac.as_ref().display());

    for (num, block) in blocks(flac)?.enumerate() {
        let block = block?;
        let block_type = block.block_type();

        println!("METADATA block #{num}");
        println!("  type: {} ({block_type})", block_type as u8);

        match block {
            Block::Streaminfo(b) => display_streaminfo(b),
            Block::Padding(b) => display_padding(b),
            Block::Application(b) => display_application(b),
            Block::SeekTable(b) => display_seektable(b),
            Block::VorbisComment(b) => display_vorbis_comment(b),
            Block::Cuesheet(b) => display_cuesheet(b),
            Block::Picture(b) => display_picture(b),
        }
    }

    Ok(())
}

fn display_streaminfo(streaminfo: Streaminfo) {
    println!(
        "  minimum blocksize: {} samples",
        streaminfo.minimum_block_size
    );
    println!(
        "  maximum blocksize: {} samples",
        streaminfo.maximum_block_size
    );
    if let Some(minimum_frame_size) = streaminfo.minimum_frame_size {
        println!("  minimum framesize: {} bytes", minimum_frame_size.get());
    }
    if let Some(maximum_frame_size) = streaminfo.maximum_frame_size {
        println!("  maximum framesize: {} bytes", maximum_frame_size.get());
    }
    println!("  sample rate: {} Hz", streaminfo.sample_rate);
    println!("  channels: {}", streaminfo.channels.get());
    println!(
        "  bits-per-sample: {}",
        u32::from(streaminfo.bits_per_sample)
    );
    if let Some(total_samples) = streaminfo.total_samples {
        println!("  total samples: {}", total_samples.get());
    }
    if let Some(md5) = streaminfo.md5 {
        println!("  MD5 signature: {}", Hex(&md5));
    }
}

fn display_padding(padding: Padding) {
    println!("  length: {}", padding.size)
}

fn display_application(application: Application) {
    println!("  length: {}", application.data.len() + 4);
    println!("  applcation ID: {:X}", application.id);
}

fn display_seektable(seektable: SeekTable) {
    use flac_codec::metadata::SeekPoint;

    println!("  seek points: {}", seektable.points.len());
    for (num, point) in seektable.points.into_iter().enumerate() {
        match point {
            SeekPoint::Defined {
                sample_offset,
                byte_offset,
                frame_samples,
            } => println!(
                "    point {num}: sample number = {}, stream offset={:X}, frame samples={}",
                sample_offset, byte_offset, frame_samples,
            ),
            SeekPoint::Placeholder => println!("    point {num}: placeholder"),
        }
    }
}

fn display_vorbis_comment(comment: VorbisComment) {
    println!("  vendor string: {}", comment.vendor_string);
    println!("  comments: {}", comment.fields.len());
    for (num, field) in comment.fields.into_iter().enumerate() {
        println!("    comment[{num}]: {field}");
    }
}

fn display_cuesheet(cuesheet: Cuesheet) {
    println!(
        "  media catalog number: {}",
        Hex(trim_nulls(cuesheet.catalog_number.as_slice()))
    );
    println!("  lead-in: {}", cuesheet.lead_in_samples);
    println!("  is CDDA: {}", cuesheet.is_cdda);
    println!("  number of tracks: {}", cuesheet.tracks.len());

    for (num, track) in cuesheet.tracks.into_iter().enumerate() {
        println!("    track[{num}]");
        println!("      offset: {}", track.offset);
        if track.number == 255 {
            println!("      number: 255 (LEAD-OUT)");
        } else {
            println!("      number: {}", track.number);
            match track.isrc {
                Some(isrc) => println!("      ISRC: {}", Hex(&isrc)),
                None => println!("      IRSC:"),
            }
            println!(
                "      type: {}",
                if track.non_audio {
                    "NON-AUDIO"
                } else {
                    "AUDIO"
                }
            );
            println!("      pre-emphasis: {}", track.pre_emphasis);
            println!("      number of index points: {}", track.index_points.len());
            for (num, point) in track.index_points.into_iter().enumerate() {
                println!("        index[{num}]");
                println!("          offset: {}", point.offset);
                println!("          number: {}", point.number);
            }
        }
    }
}

fn display_picture(picture: Picture) {
    let picture_type = picture.picture_type;
    println!("  picture type: {} ({})", picture_type as u8, picture_type);
    println!("  MIME type: {}", picture.media_type);
    println!("  description: {}", picture.description);
    println!("  width: {}", picture.width);
    println!("  height: {}", picture.height);
    println!("  depth: {}", picture.color_depth);
    match picture.colors_used {
        None => println!("  colors: 0 (unindexed)"),
        Some(colors) => println!("  colors: {colors}"),
    }
    println!("  data length: {}", picture.data.len());
}

struct Hex<'h>(&'h [u8]);

impl std::fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{:02x}", b))
    }
}

fn trim_nulls(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., 0] = s {
        s = rest
    }
    s
}
