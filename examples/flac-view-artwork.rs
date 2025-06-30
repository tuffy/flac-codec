// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::{Picture, block};

/// Views the first piece of artwork in each FLAC file, if any

fn main() {
    for flac in std::env::args_os().skip(1) {
        if let Err(err) = block::<_, Picture>(flac)
            .map_err(Error::Flac)
            .and_then(|pictures| pictures.into_iter().try_for_each(view))
        {
            eprintln!("* {err}");
        }
    }
}

fn view(picture: Picture) -> Result<(), Error> {
    viuer::print(
        &image::load_from_memory(&picture.data)?,
        &viuer::Config {
            absolute_offset: false,
            ..viuer::Config::default()
        },
    )?;
    Ok(())
}

#[derive(Debug)]
enum Error {
    Flac(flac_codec::Error),
    Image(image::ImageError),
    Viu(viuer::ViuError),
}

impl std::error::Error for Error {}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::Image(err)
    }
}

impl From<viuer::ViuError> for Error {
    fn from(err: viuer::ViuError) -> Self {
        Error::Viu(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flac(flac) => flac.fmt(f),
            Self::Image(image) => image.fmt(f),
            Self::Viu(viu) => viu.fmt(f),
        }
    }
}
