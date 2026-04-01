"""CLI entry points for the MRI harmonization benchmark pipeline."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from mri_harmonization.config import PipelineConfig
from mri_harmonization.types import HarmonizationMethod, Site

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_download(args: argparse.Namespace) -> None:
    """Download IXI T1 dataset."""
    from mri_harmonization.acquisition.download import (
        download_file,
        extract_subjects_from_tar,
        list_tar_contents,
        select_subjects_per_site,
    )

    config = PipelineConfig(
        base_dir=Path(args.base_dir),
        subjects_per_site=args.subjects_per_site,
    )

    tar_path = config.raw_dir / "IXI-T1.tar"

    # Download tar
    logger.info("Downloading IXI T1 dataset...")
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    download_file(config.ixi_t1_url, tar_path)

    # Select balanced subjects
    all_files = list_tar_contents(tar_path)
    selected = select_subjects_per_site(all_files, config.subjects_per_site)
    logger.info(f"Selected {len(selected)} subjects ({config.subjects_per_site}/site)")

    # Extract
    extract_subjects_from_tar(tar_path, config.raw_dir, selected)

    # Delete tar
    tar_path.unlink()
    logger.info("Deleted tar archive")

    # Download demographics
    demographics_path = config.data_dir / "IXI.xls"
    logger.info("Downloading IXI demographics...")
    download_file(config.ixi_demographics_url, demographics_path)

    # Build manifest
    from mri_harmonization.acquisition.manifest import build_manifest_from_directory, save_manifest
    from mri_harmonization.acquisition.demographics import load_demographics, merge_demographics

    subjects = build_manifest_from_directory(config.raw_dir)
    demographics = load_demographics(demographics_path)
    subjects = merge_demographics(subjects, demographics)
    save_manifest(subjects, config.manifest_path)
    logger.info(f"Manifest saved: {len(subjects)} subjects")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess downloaded images (N4 + brain extraction)."""
    from mri_harmonization.acquisition.manifest import load_manifest, save_manifest
    from mri_harmonization.preprocessing.pipeline import preprocess_subject
    from mri_harmonization.types import Subject

    config = PipelineConfig(base_dir=Path(args.base_dir))
    subjects = load_manifest(config.manifest_path)

    preprocessed = []
    skipped = 0
    for i, subject in enumerate(subjects):
        # Check if already preprocessed (resumable)
        brain_path = config.preprocessed_dir / subject.site.value / f"{subject.id}_T1_brain.nii.gz"
        mask_path = config.preprocessed_dir / subject.site.value / f"{subject.id}_T1_mask.nii.gz"
        if brain_path.exists() and mask_path.exists():
            preprocessed.append(Subject(
                id=subject.id, site=subject.site, image_path=brain_path,
                mask_path=mask_path, age=subject.age, sex=subject.sex,
            ))
            skipped += 1
            continue

        logger.info(f"Preprocessing {subject.id} ({i + 1}/{len(subjects)}, {skipped} skipped)")
        try:
            result = preprocess_subject(
                subject, config.preprocessed_dir, device=args.device
            )
            preprocessed.append(result)
        except Exception as e:
            logger.error(f"Failed to preprocess {subject.id}: {e}")
            continue

    save_manifest(preprocessed, config.manifest_path)
    logger.info(f"Preprocessed {len(preprocessed)}/{len(subjects)} subjects ({skipped} resumed)")

    if args.cleanup:
        shutil.rmtree(config.raw_dir, ignore_errors=True)
        logger.info("Deleted raw data directory")


def cmd_harmonize(args: argparse.Namespace) -> None:
    """Run image-level harmonization + feature extraction (fused B+C)."""
    from mri_harmonization.acquisition.manifest import load_manifest
    from mri_harmonization.features.extractor import ExtractionConfig, FeatureExtractor
    from mri_harmonization.features.io import save_feature_matrix
    from mri_harmonization.harmonization.image_level import (
        NyulHarmonizer,
        WhiteStripeHarmonizer,
        ZScoreHarmonizer,
    )

    config = PipelineConfig(base_dir=Path(args.base_dir))
    subjects = load_manifest(config.manifest_path)
    method = HarmonizationMethod(args.method)

    # Select harmonizer
    harmonizer = None
    if method == HarmonizationMethod.ZSCORE:
        harmonizer = ZScoreHarmonizer()
    elif method == HarmonizationMethod.WHITESTRIPE:
        harmonizer = WhiteStripeHarmonizer()
    elif method == HarmonizationMethod.NYUL:
        harmonizer = NyulHarmonizer()
        # Fit on all images
        logger.info("Fitting Nyul normalizer on population...")
        images = [nib.load(s.image_path) for s in subjects]
        masks = [nib.load(s.mask_path) for s in subjects if s.mask_path]
        harmonizer.fit(images, masks if masks else None)

    # Feature extraction config
    ext_config = ExtractionConfig(feature_classes=config.feature_classes)
    extractor = FeatureExtractor(ext_config)

    # Process each subject: harmonize (in memory) + extract features
    rows = []
    for i, subject in enumerate(subjects):
        logger.info(f"Processing {subject.id} [{method.value}] ({i + 1}/{len(subjects)})")

        try:
            if harmonizer is not None and subject.mask_path:
                image = nib.load(subject.image_path)
                mask = nib.load(subject.mask_path)
                normalized = harmonizer.normalize(image, mask)

                # Save temporarily for pyradiomics (needs file path)
                tmp_path = config.features_dir / f"_tmp_{subject.id}.nii.gz"
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(normalized, tmp_path)

                features = extractor.extract(tmp_path, subject.mask_path)
                tmp_path.unlink()
            else:
                # No harmonization (baseline)
                features = extractor.extract(subject.image_path, subject.mask_path)

            features["subject_id"] = subject.id
            features["site"] = subject.site.value
            rows.append(features)
        except Exception as e:
            logger.error(f"Failed on {subject.id}: {e}")
            continue

    # Save feature matrix
    df = pd.DataFrame(rows)
    output_path = config.features_dir / f"{method.value}_features.csv"
    save_feature_matrix(df, output_path)
    logger.info(f"Features saved: {output_path} ({len(rows)} subjects)")


def cmd_combat(args: argparse.Namespace) -> None:
    """Apply ComBat to feature matrices."""
    from mri_harmonization.acquisition.manifest import load_manifest
    from mri_harmonization.features.io import load_feature_matrix, save_feature_matrix
    from mri_harmonization.harmonization.feature_level import ComBatHarmonizer

    config = PipelineConfig(base_dir=Path(args.base_dir))

    # Find all feature CSVs
    feature_files = sorted(config.features_dir.glob("*_features.csv"))
    if not feature_files:
        logger.error("No feature files found in %s", config.features_dir)
        return

    # Determine covariates from manifest
    continuous_cols = []
    categorical_cols = []
    subjects = load_manifest(config.manifest_path)
    has_demographics = any(s.age is not None for s in subjects)

    for feat_path in feature_files:
        condition = feat_path.stem.replace("_features", "")
        logger.info(f"Applying ComBat to {condition}")

        df = load_feature_matrix(feat_path)

        # Merge demographics if available
        if has_demographics:
            demo_map = {s.id: s for s in subjects}
            df["age"] = df["subject_id"].map(lambda x: demo_map.get(x, None) and demo_map[x].age)
            df["sex"] = df["subject_id"].map(lambda x: demo_map.get(x, None) and demo_map[x].sex)

            # Drop rows with missing demographics
            before = len(df)
            df = df.dropna(subset=["age", "sex"])
            if len(df) < before:
                logger.warning(f"Dropped {before - len(df)} subjects with missing demographics")

            continuous_cols = ["age"]
            categorical_cols = ["sex"]

        harmonizer = ComBatHarmonizer(
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )

        try:
            harmonized = harmonizer.harmonize(df)
            # Remove covariate columns from output
            for col in continuous_cols + categorical_cols:
                if col in harmonized.columns:
                    harmonized = harmonized.drop(columns=[col])

            output_path = config.features_dir / f"{condition}_combat_features.csv"
            save_feature_matrix(harmonized, output_path)
            logger.info(f"ComBat features saved: {output_path}")
        except Exception as e:
            logger.error(f"ComBat failed for {condition}: {e}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run reproducibility analysis."""
    from mri_harmonization.analysis.summary import analyze_reproducibility, summarize_conditions
    from mri_harmonization.features.io import load_feature_matrix, save_feature_matrix

    config = PipelineConfig(base_dir=Path(args.base_dir))

    feature_files = sorted(config.features_dir.glob("*_features.csv"))
    if not feature_files:
        logger.error("No feature files found")
        return

    all_results = {}
    for feat_path in feature_files:
        condition = feat_path.stem.replace("_features", "")
        logger.info(f"Analyzing {condition}")

        df = load_feature_matrix(feat_path)
        results = analyze_reproducibility(
            df, icc_threshold=config.icc_threshold, alpha=config.alpha
        )
        all_results[condition] = results

        # Save per-condition results
        results_df = pd.DataFrame([
            {
                "feature": r.feature_name,
                "icc": r.icc,
                "cv": r.cv,
                "kw_statistic": r.kw_statistic,
                "kw_pvalue": r.kw_pvalue,
                "is_reproducible": r.is_reproducible,
            }
            for r in results
        ])
        output = config.results_dir / f"{condition}_metrics.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output, index=False)

    # Summary across conditions
    summary = summarize_conditions(all_results)
    summary_path = config.results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved: {summary_path}")
    print("\n" + summary.to_string(index=False))


def cmd_visualize(args: argparse.Namespace) -> None:
    """Generate visualization figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mri_harmonization.acquisition.manifest import load_manifest
    from mri_harmonization.features.io import load_feature_matrix
    from mri_harmonization.visualization.bar_charts import plot_reproducibility_summary
    from mri_harmonization.visualization.distributions import plot_feature_distributions
    from mri_harmonization.visualization.heatmaps import plot_icc_heatmap
    from mri_harmonization.visualization.histograms import plot_intensity_histograms

    config = PipelineConfig(base_dir=Path(args.base_dir))
    figures_dir = config.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary bar chart
    summary_path = config.results_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        fig = plot_reproducibility_summary(summary)
        fig.savefig(figures_dir / "reproducibility_summary.png", dpi=150)
        plt.close(fig)
        logger.info("Saved reproducibility_summary.png")

    # 2. ICC heatmap
    metrics_files = sorted(config.results_dir.glob("*_metrics.csv"))
    if metrics_files:
        icc_data = {}
        for mf in metrics_files:
            condition = mf.stem.replace("_metrics", "")
            metrics = pd.read_csv(mf)
            icc_data[condition] = metrics.set_index("feature")["icc"]

        icc_df = pd.DataFrame(icc_data)
        if len(icc_df) > 30:
            icc_df = icc_df.loc[icc_df.std(axis=1).nlargest(30).index]

        fig = plot_icc_heatmap(icc_df)
        fig.savefig(figures_dir / "icc_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved icc_heatmap.png")

    # 3. Intensity histograms (before/after each method)
    subjects = load_manifest(config.manifest_path)
    if subjects and subjects[0].image_path.exists():
        _generate_intensity_histograms(subjects, config, figures_dir)

    # 4. Post-harmonization intensity histograms
    if subjects and subjects[0].image_path.exists():
        _generate_post_harmonization_histograms(subjects, figures_dir)

    # 5. Feature distribution violin plots (before/after)
    _generate_feature_violin_plots(config, figures_dir)

    logger.info(f"Figures saved to {figures_dir}")


def _generate_intensity_histograms(
    subjects: list,
    config: "PipelineConfig",
    figures_dir: Path,
) -> None:
    """Generate per-site intensity histograms."""
    import matplotlib.pyplot as plt

    from mri_harmonization.visualization.histograms import plot_intensity_histograms

    # Sample voxel intensities from preprocessed images
    intensity_data: dict[str, np.ndarray] = {}
    for subject in subjects:
        if not subject.image_path.exists() or subject.mask_path is None:
            continue
        img = nib.load(subject.image_path)
        mask = nib.load(subject.mask_path)
        data = img.get_fdata()
        mask_data = mask.get_fdata().astype(bool)
        brain_values = data[mask_data]
        # Subsample to keep memory manageable
        rng = np.random.RandomState(hash(subject.id) % 2**31)
        sample = rng.choice(brain_values, size=min(5000, len(brain_values)), replace=False)
        site = subject.site.value
        if site not in intensity_data:
            intensity_data[site] = sample
        else:
            intensity_data[site] = np.concatenate([intensity_data[site], sample])

    if intensity_data:
        fig = plot_intensity_histograms(
            intensity_data, title="Preprocessed Intensity Distributions by Site"
        )
        fig.savefig(figures_dir / "intensity_histograms.png", dpi=150)
        plt.close(fig)
        logger.info("Saved intensity_histograms.png")


def _generate_post_harmonization_histograms(
    subjects: list,
    figures_dir: Path,
) -> None:
    """Generate intensity histograms after each normalization method."""
    import matplotlib.pyplot as plt

    from mri_harmonization.harmonization.image_level import (
        NyulHarmonizer,
        WhiteStripeHarmonizer,
        ZScoreHarmonizer,
    )
    from mri_harmonization.visualization.histograms import plot_intensity_histograms

    # Load all images and masks once
    loaded = []
    for s in subjects:
        if not s.image_path.exists() or s.mask_path is None:
            continue
        loaded.append((s, nib.load(s.image_path), nib.load(s.mask_path)))

    if not loaded:
        return

    def _sample_intensities(
        image: "nib.Nifti1Image", mask: "nib.Nifti1Image", site: str,
        data_dict: dict[str, np.ndarray],
    ) -> None:
        d = image.get_fdata()
        m = mask.get_fdata().astype(bool)
        vals = d[m]
        rng = np.random.RandomState(42)
        sample = rng.choice(vals, size=min(5000, len(vals)), replace=False)
        if site not in data_dict:
            data_dict[site] = sample
        else:
            data_dict[site] = np.concatenate([data_dict[site], sample])

    methods = [
        ("Z-score", ZScoreHarmonizer()),
        ("WhiteStripe", WhiteStripeHarmonizer()),
    ]

    # Nyul needs fitting first
    nyul = NyulHarmonizer()
    images = [img for _, img, _ in loaded]
    masks = [msk for _, _, msk in loaded]
    nyul.fit(images, masks)
    methods.append(("Nyul", nyul))

    for method_name, harmonizer in methods:
        intensity_data: dict[str, np.ndarray] = {}
        for subject, img, mask in loaded:
            try:
                normalized = harmonizer.normalize(img, mask)
                _sample_intensities(normalized, mask, subject.site.value, intensity_data)
            except Exception as e:
                logger.warning(f"{method_name} failed on {subject.id}: {e}")
                continue

        if intensity_data:
            fig = plot_intensity_histograms(
                intensity_data,
                title=f"Intensity Distributions After {method_name} Normalization",
            )
            fname = f"intensity_histograms_{method_name.lower().replace('-', '')}.png"
            fig.savefig(figures_dir / fname, dpi=150)
            plt.close(fig)
            logger.info(f"Saved {fname}")


def _generate_feature_violin_plots(config: "PipelineConfig", figures_dir: Path) -> None:
    """Generate violin plots comparing features before/after harmonization."""
    import matplotlib.pyplot as plt

    from mri_harmonization.features.io import load_feature_matrix
    from mri_harmonization.visualization.distributions import plot_feature_distributions

    # Pick representative features: one firstorder, one texture
    representative_features = []
    none_path = config.features_dir / "none_features.csv"
    if not none_path.exists():
        return
    df = load_feature_matrix(none_path)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "site")]

    # Find a firstorder and a glcm feature
    for col in feature_cols:
        if "firstorder" in col.lower() and "mean" in col.lower():
            representative_features.append(col)
            break
    for col in feature_cols:
        if "glcm" in col.lower() and "correlation" in col.lower():
            representative_features.append(col)
            break
    if not representative_features:
        representative_features = feature_cols[:2]

    # Plot for each condition pair: raw vs combat
    conditions = ["none", "zscore", "whitestripe", "nyul"]
    for cond in conditions:
        raw_path = config.features_dir / f"{cond}_features.csv"
        combat_path = config.features_dir / f"{cond}_combat_features.csv"

        if not raw_path.exists():
            continue

        raw_df = load_feature_matrix(raw_path)

        # Raw features violin
        fig = plot_feature_distributions(
            raw_df, representative_features,
            title=f"Feature Distributions: {cond} (before ComBat)"
        )
        fig.savefig(figures_dir / f"violin_{cond}_raw.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Combat features violin
        if combat_path.exists():
            combat_df = load_feature_matrix(combat_path)
            fig = plot_feature_distributions(
                combat_df, representative_features,
                title=f"Feature Distributions: {cond} + ComBat"
            )
            fig.savefig(figures_dir / f"violin_{cond}_combat.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    logger.info("Saved violin plot figures")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MRI Harmonization Benchmark for Radiomics Reproducibility"
    )
    parser.add_argument(
        "--base-dir", default=".", help="Project base directory (default: .)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage")

    # Download
    dl = subparsers.add_parser("download", help="Download IXI T1 dataset")
    dl.add_argument("--subjects-per-site", type=int, default=50)

    # Preprocess
    pp = subparsers.add_parser("preprocess", help="N4 + brain extraction")
    pp.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    pp.add_argument("--cleanup", action="store_true", help="Delete raw data after")

    # Harmonize
    hz = subparsers.add_parser("harmonize", help="Image harmonization + feature extraction")
    hz.add_argument(
        "--method",
        required=True,
        choices=[m.value for m in HarmonizationMethod],
    )

    # ComBat
    subparsers.add_parser("combat", help="Apply ComBat to feature matrices")

    # Analyze
    subparsers.add_parser("analyze", help="Reproducibility analysis")

    # Visualize
    subparsers.add_parser("visualize", help="Generate figures")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "harmonize": cmd_harmonize,
        "combat": cmd_combat,
        "analyze": cmd_analyze,
        "visualize": cmd_visualize,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
