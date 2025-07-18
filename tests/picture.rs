// Copyright 2025 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use flac_codec::metadata::{Picture, PictureType};

#[test]
fn test_png() {
    fn test(width: u32, height: u32, color_depth: u32, colors_used: u32, data: &[u8]) {
        use std::num::NonZero;

        assert_eq!(
            Picture::new(PictureType::FrontCover, String::new(), Vec::from(data)).unwrap(),
            Picture {
                picture_type: PictureType::FrontCover,
                media_type: "image/png".to_owned(),
                description: String::new(),
                width,
                height,
                color_depth,
                colors_used: NonZero::new(colors_used),
                data: Vec::from(data),
            }
        );
    }

    test(32, 32, 1, 0, include_bytes!("data/images/basn0g01.png"));
    test(32, 32, 2, 0, include_bytes!("data/images/basn0g02.png"));
    test(32, 32, 4, 0, include_bytes!("data/images/basn0g04.png"));
    test(32, 32, 8, 0, include_bytes!("data/images/basn0g08.png"));
    test(32, 32, 16, 0, include_bytes!("data/images/basn0g16.png"));
    test(32, 32, 24, 0, include_bytes!("data/images/basn2c08.png"));
    test(32, 32, 48, 0, include_bytes!("data/images/basn2c16.png"));
    test(
        32,
        32,
        0,
        1 << 1,
        include_bytes!("data/images/basn3p01.png"),
    );
    test(
        32,
        32,
        0,
        1 << 2,
        include_bytes!("data/images/basn3p02.png"),
    );
    test(
        32,
        32,
        0,
        1 << 8,
        include_bytes!("data/images/basn3p08.png"),
    );
    test(32, 32, 16, 0, include_bytes!("data/images/basn4a08.png"));
    test(32, 32, 32, 0, include_bytes!("data/images/basn4a16.png"));
    test(32, 32, 32, 0, include_bytes!("data/images/basn6a08.png"));
    test(32, 32, 64, 0, include_bytes!("data/images/basn6a16.png"));
    test(8, 8, 24, 0, include_bytes!("data/images/cdsn2c08.png"));
}

#[test]
fn test_jpeg() {
    fn test(width: u32, height: u32, color_depth: u32, data: &[u8]) {
        assert_eq!(
            Picture::new(PictureType::FrontCover, String::new(), Vec::from(data)).unwrap(),
            Picture {
                picture_type: PictureType::FrontCover,
                media_type: "image/jpeg".to_owned(),
                description: String::new(),
                width,
                height,
                color_depth,
                colors_used: None,
                data: Vec::from(data),
            }
        );
    }

    test(
        49,
        37,
        24,
        include_bytes!("data/images/sample-blue-100x75.jpg"),
    );
    test(
        100,
        100,
        24,
        include_bytes!("data/images/sample-green-200x200.jpg"),
    );
    test(
        150,
        150,
        24,
        include_bytes!("data/images/sample-red-400x300.jpg"),
    );
}

#[test]
fn test_gif() {
    fn test(width: u32, height: u32, colors_used: u32, data: &[u8]) {
        use std::num::NonZero;

        assert_eq!(
            Picture::new(PictureType::FrontCover, String::new(), Vec::from(data)).unwrap(),
            Picture {
                picture_type: PictureType::FrontCover,
                media_type: "image/gif".to_owned(),
                description: String::new(),
                width,
                height,
                color_depth: 0,
                colors_used: NonZero::new(colors_used),
                data: Vec::from(data),
            }
        );
    }

    test(
        50,
        38,
        2,
        include_bytes!("data/images/sample-red-100x75.gif"),
    );
    test(
        100,
        100,
        2,
        include_bytes!("data/images/sample-green-200x200.gif"),
    );
    test(
        200,
        150,
        2,
        include_bytes!("data/images/sample-blue-400x300.gif"),
    );
}
