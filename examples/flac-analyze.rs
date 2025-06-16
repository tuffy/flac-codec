use flac_codec::{
    Error,
    stream::{Frame, Residuals, SignedInteger, Subframe},
};
use std::path::Path;

/// This corresponds to the reference implementation's:
///
/// "flac -a <file.flac>"
///
/// In that it prints a full analysis dump of the input file.
/// Residual text is omitted because that's a *lot* of information
/// to display all at once.

fn main() {
    match std::env::args_os().skip(1).next() {
        Some(file) => {
            if let Err(err) = display_analysis(&file) {
                eprintln!("* {}, {err}", file.display());
            }
        }
        None => eprintln!("* Usage: flac-analyze <file.flac>"),
    }
}

fn display_analysis<P: AsRef<Path>>(path: P) -> Result<(), Error> {
    use flac_codec::stream::FrameIterator;

    // used to determine the size of the last frame
    let file_len = path.as_ref().metadata().map(|m| m.len())?;
    let mut frames = FrameIterator::open(path)?.peekable();

    while let Some((frame, offset)) = frames.next().transpose()? {
        // we don't keep track of each frame's size in bytes,
        // but we can derive that information by peeking ahead
        // to the next frame's offset
        display_frame(
            match frames.peek() {
                Some(Ok((_, next_offset))) => next_offset - offset,
                Some(Err(_)) => 0, // next frame is an error
                None => file_len - offset,
            },
            offset,
            frame,
        )
    }

    Ok(())
}

fn display_frame(len: u64, offset: u64, frame: Frame) {
    use flac_codec::stream::SubframeWidth;

    println!(
        "frame={}\toffset={offset}\tbits={}\tblocksize={}\tsample_rate={}\tchannels={}\tchannel_assignment={}",
        frame.header.frame_number,
        len * 8,
        frame.header.block_size,
        frame.header.sample_rate,
        frame.header.channel_assignment.count(),
        frame.header.channel_assignment,
    );

    for (num, subframe) in frame.subframes.into_iter().enumerate() {
        match subframe {
            SubframeWidth::Common(s) => display_subframe(num, s),
            SubframeWidth::Wide(s) => display_subframe(num, s),
        }
    }
}

fn display_subframe<I: SignedInteger>(num: usize, subframe: Subframe<I>) {
    let subframe_type = subframe.subframe_type();
    match subframe {
        Subframe::Constant {
            wasted_bps, sample, ..
        } => println!(
            "\tsubframe={num}\twasted_bits={wasted_bps}\ttype={subframe_type}\tvalue={sample}"
        ),
        Subframe::Verbatim { wasted_bps, .. } => {
            println!("\tsubframe={num}\twasted_bits={wasted_bps}\ttype={subframe_type}")
        }
        Subframe::Fixed {
            order,
            warm_up,
            wasted_bps,
            residuals,
        } => {
            println!(
                "\tsubframe={num}\twasted_bits={wasted_bps}\ttype={subframe_type}\torder={order}\tresiduals_type={}\tpartition_order={}",
                residuals.coding_method(),
                residuals.partition_order()
            );
            for (num, warm_up) in warm_up.into_iter().enumerate() {
                println!("\t\twarmup[{num}]={warm_up}");
            }
            display_residuals(residuals);
        }
        Subframe::Lpc {
            order,
            warm_up,
            precision,
            shift,
            coefficients,
            residuals,
            wasted_bps,
        } => {
            println!(
                "\tsubframe={num}\twasted_bits={wasted_bps}\ttype={subframe_type}\torder={order}\tqlp_coeff_precision={}\tquantization_level={shift}\tresiduals_type={}\tpartition_order={}",
                u32::from(precision),
                residuals.coding_method(),
                residuals.partition_order()
            );
            for (num, coeff) in coefficients.into_iter().enumerate() {
                println!("\t\tqlp_coeff[{num}]={coeff}");
            }
            for (num, warm_up) in warm_up.into_iter().enumerate() {
                println!("\t\twarmup[{num}]={warm_up}");
            }
            display_residuals(residuals);
        }
    }
}

fn display_residuals<I: SignedInteger>(residuals: Residuals<I>) {
    use flac_codec::stream::ResidualPartition;

    fn display_residual_partition<const RICE_MAX: u32, I>(
        num: usize,
        partition: ResidualPartition<RICE_MAX, I>,
    ) where
        I: SignedInteger,
    {
        match partition {
            ResidualPartition::Standard { rice, .. } => {
                println!("\t\tparameter[{num}]={}", u32::from(rice))
            }
            ResidualPartition::Escaped { escape_size, .. } => println!(
                "\t\tparameter[{num}]=ESCAPE, raw_bits={}",
                u32::from(escape_size)
            ),
            ResidualPartition::Constant { .. } => {
                println!("\t\tparameter[{num}]=ESCAPE, raw_bits=0")
            }
        }
    }

    match residuals {
        Residuals::Method0 { partitions } => partitions
            .into_iter()
            .enumerate()
            .for_each(|(num, p)| display_residual_partition(num, p)),
        Residuals::Method1 { partitions } => partitions
            .into_iter()
            .enumerate()
            .for_each(|(num, p)| display_residual_partition(num, p)),
    }
}
