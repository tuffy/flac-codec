use flac_codec::decode::verify;
use std::path::{Path, PathBuf};

fn corpus_dir(group: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("flac-test-files")
        .join(group)
}

fn corpus_files(group: &str) -> Vec<PathBuf> {
    let directory = corpus_dir(group);
    let mut files = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("unable to read {}: {error}", directory.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "flac")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn file_number(path: &Path) -> u8 {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.get(..2))
        .and_then(|number| number.parse().ok())
        .unwrap_or_else(|| panic!("invalid corpus filename: {}", path.display()))
}

#[test]
#[ignore = "requires the external flac-test-files decoder corpus"]
fn test_subset_corpus() {
    let files = corpus_files("subset");
    assert_eq!(files.len(), 64);

    for file in files {
        assert!(
            verify(&file).is_ok(),
            "valid subset file failed verification: {}",
            file.display()
        );
    }
}

#[test]
#[ignore = "requires the external flac-test-files decoder corpus"]
fn test_uncommon_corpus() {
    let files = corpus_files("uncommon");
    assert_eq!(files.len(), 11);

    for file in files {
        let supported = matches!(file_number(&file), 5..=9);
        assert_eq!(
            verify(&file).is_ok(),
            supported,
            "unexpected uncommon-file result: {}",
            file.display()
        );
    }
}

#[test]
#[ignore = "requires the external flac-test-files decoder corpus"]
fn test_faulty_corpus() {
    let files = corpus_files("faulty");
    assert_eq!(files.len(), 11);

    for file in files {
        assert!(
            verify(&file).is_err(),
            "faulty file passed verification: {}",
            file.display()
        );
    }
}
