// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::{VorbisComment, block, update};

/// Roughly corresponds to the reference implementation's:
///
/// "metaflac --set-tag=FIELD <file.flac>"
///
/// Though without a full argument parser,
/// it operates on only a single FLAC file at time.
/// Also, when no tags are indicated, it simply
/// display's the file's existing tags.

fn main() {
    match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [flac] => match block::<_, VorbisComment>(flac) {
            Ok(Some(comment)) => {
                for tag in comment.fields {
                    println!("{tag}");
                }
            }
            Ok(None) => {
                // no VorbisComment, so nothing to display
            }
            Err(err) => eprintln!("* Error: {err}"),
        },
        [tags @ .., flac] if tags.iter().all(|t| t.contains('=')) => {
            match update(&flac, |blocklist| {
                blocklist.update::<VorbisComment>(|vorbis_comment| {
                    for tag in tags {
                        if let Some((field, value)) = tag.split_once('=') {
                            vorbis_comment.set(field, value);
                        }
                    }
                });
                Ok::<(), flac_codec::Error>(())
            }) {
                Ok(_) => println!("* {flac}: Updated"),
                Err(err) => println!("* Error: {flac} - {err}"),
            }
        }
        _ => eprintln!("* Usage: \"TITLE=Track Title\" \"ALBUM=Album Title\" <file.flac>"),
    }
}
