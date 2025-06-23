// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::{Picture, PictureType, update};

/// Roughly corresponds to the reference implementation's:
///
/// "metaflac --import-picture-from=<file.png> <file.flac> <file2.flac> ..."
///
/// Though without a full argument parser,
/// it operates on only a single cover image.
/// Also, when no FLAC files are indicated, it simply
/// display's the image's file metrics.

fn main() {
    match std::env::args_os().skip(1).collect::<Vec<_>>().as_slice() {
        [image] => match Picture::open(PictureType::FrontCover, "", image) {
            Ok(Picture {
                media_type,
                width,
                height,
                color_depth,
                colors_used,
                ..
            }) => {
                println!("  MIME type : {media_type}");
                println!("      width : {width}");
                println!("     height : {height}");
                println!("color depth : {color_depth}");
                if let Some(colors_used) = colors_used {
                    println!("colors used : {colors_used}");
                }
            }
            Err(err) => eprintln!("* {} : {err}", image.display()),
        },
        [image, flacs @ ..] => match Picture::open(PictureType::FrontCover, "", image) {
            Ok(picture) => {
                if let Err(err) = flacs.iter().try_for_each(|flac| {
                    update(flac, |blocklist| {
                        // remove any existing picture
                        blocklist.remove::<Picture>();
                        // add our new front cover
                        blocklist.insert(picture.clone());
                        Ok::<(), flac_codec::Error>(())
                    })
                    .map(|_| ())
                }) {
                    eprintln!("* Error : {err}");
                }
            }
            Err(err) => eprintln!("* {} : {err}", image.display()),
        },
        [] => eprintln!("* Usage: <image file> <file1.flac> <file2.flac> ..."),
    }
}
