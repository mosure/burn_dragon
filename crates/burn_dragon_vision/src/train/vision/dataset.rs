use crate::train::prelude::*;

pub(crate) fn maybe_download_vision_dataset(config: &VisionDatasetConfig) -> Result<()> {
    match config.source {
        VisionDatasetSource::Imagenet => {
            let Some(download) = &config.download else {
                return Ok(());
            };

            let train_root = config.imagenet_root.join(&config.train_dir);
            let val_root = config.imagenet_root.join(&config.val_dir);
            if vision_split_has_images(&train_root)? && vision_split_has_images(&val_root)? {
                return Ok(());
            }

            match download {
                VisionDatasetDownloadConfig::Imagenette { variant } => {
                    download_imagenette(config, *variant)
                }
                VisionDatasetDownloadConfig::Cifar => Err(anyhow!(
                    "dataset.download.type = \"cifar\" requires dataset.source = \"cifar10\" or \"cifar100\""
                )),
            }
        }
        VisionDatasetSource::MovingMnist => {
            MovingMnistVideoDataset::ensure_mnist_downloaded();
            Ok(())
        }
        VisionDatasetSource::Cifar10 | VisionDatasetSource::Cifar100 => {
            let Some(download) = &config.download else {
                return Ok(());
            };
            let cifar_type = match config.source {
                VisionDatasetSource::Cifar10 => CifarType::Cifar10,
                VisionDatasetSource::Cifar100 => CifarType::Cifar100,
                _ => unreachable!("guarded by match"),
            };
            if cifar_root_has_records(&config.cifar_root, cifar_type)? {
                return Ok(());
            }
            match download {
                VisionDatasetDownloadConfig::Cifar => download_cifar(config, cifar_type),
                VisionDatasetDownloadConfig::Imagenette { .. } => Err(anyhow!(
                    "dataset.download.type = \"imagenette\" requires dataset.source = \"imagenet\""
                )),
            }
        }
    }
}

pub(crate) fn vision_split_has_images(root: &Path) -> Result<bool> {
    if !root.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        for file in
            fs::read_dir(&path).with_context(|| format!("failed to read {}", path.display()))?
        {
            let file = file?;
            let path = file.path();
            if path.is_file() && is_image_file(&path) {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

pub(crate) fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"),
        None => false,
    }
}

pub(crate) fn download_imagenette(
    config: &VisionDatasetConfig,
    variant: ImagenetteVariant,
) -> Result<()> {
    if config.train_dir != "train" || config.val_dir != "val" {
        return Err(anyhow!(
            "imagenette download expects train_dir='train' and val_dir='val'"
        ));
    }

    let root = &config.imagenet_root;
    let train_root = root.join(&config.train_dir);
    let val_root = root.join(&config.val_dir);
    if vision_split_has_images(&train_root)? && vision_split_has_images(&val_root)? {
        return Ok(());
    }
    if root.exists() {
        let has_entries = fs::read_dir(root)
            .map(|mut iter| iter.next().is_some())
            .unwrap_or(false);
        if has_entries {
            return Err(anyhow!(
                "imagenet_root {} exists but doesn't look like imagenette; move it or disable download",
                root.display()
            ));
        }
    }

    if let Some(parent) = root.parent() {
        fs::create_dir_all(parent)?;
    }

    let cache_root = root.parent().unwrap_or_else(|| Path::new("."));
    let folder = imagenette_folder_name(variant);
    let cache_dir = cache_root.join(".vision_cache").join(folder);
    let extract_dir = download_and_extract_tgz(
        imagenette_url(variant),
        &format!("{folder}.tgz"),
        &cache_dir,
    )?;

    let candidate = extract_dir.join(folder);
    let source_dir = if candidate.is_dir() {
        candidate
    } else if extract_dir.join(&config.train_dir).is_dir() {
        extract_dir.clone()
    } else {
        return Err(anyhow!(
            "unexpected imagenette archive layout under {}",
            extract_dir.display()
        ));
    };

    if root.exists() {
        let has_entries = fs::read_dir(root)
            .map(|mut iter| iter.next().is_some())
            .unwrap_or(false);
        if has_entries {
            return Err(anyhow!(
                "imagenet_root {} exists but is not empty",
                root.display()
            ));
        }
        fs::remove_dir_all(root)?;
    }

    if let Err(err) = fs::rename(&source_dir, root) {
        copy_dir_all(&source_dir, root).map_err(|copy_err| {
            anyhow!(
                "failed to move imagenette data into {}: {err}; copy error: {copy_err}",
                root.display()
            )
        })?;
    }

    Ok(())
}

pub(crate) fn download_cifar(config: &VisionDatasetConfig, cifar_type: CifarType) -> Result<()> {
    let root = &config.cifar_root;
    let folder = cifar_folder_name(cifar_type);
    let target_dir = root.join(folder);
    if cifar_root_has_records(root, cifar_type)? {
        return Ok(());
    }

    fs::create_dir_all(root)?;
    let cache_root = root.parent().unwrap_or_else(|| Path::new("."));
    let cache_dir = cache_root.join(".vision_cache").join(folder);
    let extract_dir = download_and_extract_tgz(
        cifar_url(cifar_type),
        cifar_archive_name(cifar_type),
        &cache_dir,
    )?;

    let source_dir = extract_dir.join(folder);
    if !source_dir.is_dir() {
        return Err(anyhow!(
            "unexpected cifar archive layout under {}",
            extract_dir.display()
        ));
    }
    replace_directory(&source_dir, &target_dir).map_err(|err| {
        anyhow!(
            "failed to move CIFAR data into {}: {err}",
            target_dir.display()
        )
    })?;
    Ok(())
}

pub(crate) fn imagenette_url(variant: ImagenetteVariant) -> &'static str {
    match variant {
        ImagenetteVariant::Imagenette2_160 => {
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
        }
        ImagenetteVariant::Imagenette2_320 => {
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        }
    }
}

pub(crate) fn cifar_url(cifar_type: CifarType) -> &'static str {
    match cifar_type {
        CifarType::Cifar10 => "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        CifarType::Cifar100 => "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
    }
}

pub(crate) fn imagenette_folder_name(variant: ImagenetteVariant) -> &'static str {
    match variant {
        ImagenetteVariant::Imagenette2_160 => "imagenette2-160",
        ImagenetteVariant::Imagenette2_320 => "imagenette2-320",
    }
}

pub(crate) fn cifar_folder_name(cifar_type: CifarType) -> &'static str {
    match cifar_type {
        CifarType::Cifar10 => "cifar-10-batches-bin",
        CifarType::Cifar100 => "cifar-100-binary",
    }
}

fn cifar_archive_name(cifar_type: CifarType) -> &'static str {
    match cifar_type {
        CifarType::Cifar10 => "cifar-10-binary.tar.gz",
        CifarType::Cifar100 => "cifar-100-binary.tar.gz",
    }
}

pub(crate) fn cifar_root_has_records(root: &Path, cifar_type: CifarType) -> Result<bool> {
    let resolved = root.join(cifar_folder_name(cifar_type));
    let required = match cifar_type {
        CifarType::Cifar10 => vec![
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
            "test_batch.bin",
        ],
        CifarType::Cifar100 => vec!["train.bin", "test.bin"],
    };
    if !resolved.is_dir() {
        return Ok(false);
    }
    Ok(required
        .into_iter()
        .all(|name| resolved.join(name).is_file()))
}

fn download_and_extract_tgz(url: &str, archive_name: &str, cache_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir)?;
    let archive_path = cache_dir.join(archive_name);
    if !archive_path.is_file() {
        download_file(url, &archive_path)?;
    }
    let extract_dir = cache_dir.join("extract");
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir)?;
    }
    fs::create_dir_all(&extract_dir)?;
    unpack_tgz_archive(&archive_path, &extract_dir)?;
    Ok(extract_dir)
}

fn unpack_tgz_archive(archive_path: &Path, extract_dir: &Path) -> Result<()> {
    let archive_file = fs::File::open(archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let decoder = flate2::read::GzDecoder::new(archive_file);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(extract_dir)
        .with_context(|| format!("failed to unpack {}", archive_path.display()))?;
    Ok(())
}

fn replace_directory(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Err(err) = fs::rename(src, dst) {
        copy_dir_all(src, dst).map_err(|copy_err| {
            anyhow!(
                "rename failed ({err}); copy into {} also failed: {copy_err}",
                dst.display()
            )
        })?;
    }
    Ok(())
}

pub(crate) fn download_file(url: &str, dest: &Path) -> Result<()> {
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow!("download destination missing parent"))?;
    fs::create_dir_all(parent)?;

    info!("Downloading {url}");
    let response = ureq::get(url)
        .call()
        .map_err(|err| anyhow!("failed to download {url}: {err}"))?;
    let mut reader = response.into_reader();
    let tmp_path = dest.with_extension("tmp");
    let mut file = fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    fs::rename(&tmp_path, dest).with_context(|| {
        format!(
            "failed to rename {} to {}",
            tmp_path.display(),
            dest.display()
        )
    })?;
    Ok(())
}

pub(crate) fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src).with_context(|| format!("failed to read {}", src.display()))? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }
    Ok(())
}
