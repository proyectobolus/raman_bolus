"""RamanBatch v2 - Refactored batch processing class."""
from .batch_loader import (
    load_all_spectra, load_excluded_times, load_included_times, deduplicate_and_sort_spectra
)
from .batch_processing import (
    reshape_all_spectra, interpolate_all_spectra, check_common_xrange,
    apply_baseline_to_batch, get_baseline_statistics, print_baseline_summary,
    export_acquired_times, plot_all_raw_spectra, plot_cleaned_spectra
)


class RamanBatch:
    """Batch processing of Raman spectra (v2 - Modular)."""

    def __init__(self, root='.', avoided_dirs=None, file_suffix='.txt',
                 first_line_prefix="#Acq. time (s)=", fit_common_xrange=False,
                 num_points=2000, excluded_acquired_times=None, avoid_file=None,
                 range_limits=(100, 1000), search_dirs=None,
                 included_acquired_times=None, include_file=None):
        """Initialize RamanBatch.

        Parameters
        ----------
        root : str
            Single root directory to search (used if `search_dirs` is None)
        search_dirs : list[str] | None
            Optional list of directories to search; overrides `root` when provided
        range_limits : tuple[int, int]
            Default x-range to reshape spectra during initialization (100, 1000)
        avoid_file : str | None
            Path to file containing acquisition times to exclude
        include_file : str | None
            Path to file containing acquisition times to include; when provided,
            only spectra with these acquisition times will be loaded
        """
        print(f"RamanBatch: Initializing with root: {root}")

        self.root = root
        self.search_dirs = list(search_dirs) if search_dirs else None
        self.range_limits = range_limits
        self.avoided_dirs = set(avoided_dirs or [])
        self.file_suffix = file_suffix
        self.first_line_prefix = first_line_prefix
        self.all_raman = {}
        self.counter = 0
        self.already_acquired = []
        self.common_xrange = None
        self.fit_common_xrange = fit_common_xrange
        self.num_points = num_points
        self.baseline_results = None

        # Load excluded times
        if avoid_file is not None:
            self.excluded_acquired_times = load_excluded_times(avoid_file)
        else:
            self.excluded_acquired_times = set(excluded_acquired_times or [])

        # Load included times
        if include_file is not None:
            self.included_acquired_times = load_included_times(include_file)
        else:
            self.included_acquired_times = set(included_acquired_times or []) if included_acquired_times else None

        # Load and process
        self._load_and_process()
    
    def __len__(self):
        return self.counter
    
    def _load_and_process(self):
        """Load, deduplicate, and process spectra."""
        # Load (supports single root or multiple search directories)
        self.all_raman = load_all_spectra(
            self.root,
            avoided_dirs=self.avoided_dirs,
            file_suffix=self.file_suffix,
            first_line_prefix=self.first_line_prefix,
            excluded_times=self.excluded_acquired_times,
            roots=self.search_dirs,
            included_times=self.included_acquired_times
        )
        self.counter = len(self.all_raman)
        
        if self.counter > 0:
            # Deduplicate and sort
            self.all_raman, self.already_acquired = deduplicate_and_sort_spectra(self.all_raman)
            self.counter = len(self.all_raman)
            
            print(f"Assigned {len(self.all_raman)} batch IDs in chronological order")
            print(f"  Earliest: {self.already_acquired[0] if self.already_acquired else 'N/A'}")
            print(f"  Latest: {self.already_acquired[-1] if self.already_acquired else 'N/A'}")
            
            # Reshape to provided range during initialization
            if self.range_limits is not None:
                print(f"Reshaping all spectra to range: {self.range_limits}")
                self.reshape_all(self.range_limits)
            else:
                # Fallback: attempt common range if requested
                try:
                    common_xmin, common_xmax = check_common_xrange(self.all_raman)
                    self.common_xrange = (common_xmin, common_xmax)

                    if self.fit_common_xrange:
                        print("Fitting window as fit_common_xrange is True")
                        self.reshape_all(self.common_xrange)
                except ValueError as e:
                    print(f"Warning: {e}")
            
            self._print_attribute_info()
    
    def reshape_all(self, range_limits, verbose=False):
        """Apply range reshaping to all spectra."""
        reshape_all_spectra(self.all_raman, range_limits, verbose=verbose)
    
    def interpolate_all(self, final_size):
        """Apply interpolation to all spectra."""
        interpolate_all_spectra(self.all_raman, final_size)
    
    def baseline_on_data(self, methods=None, verbose=True, **kwargs):
        """Apply baseline correction to all spectra."""
        self.baseline_results = apply_baseline_to_batch(
            self.all_raman, methods=methods, verbose=verbose, **kwargs
        )
        return self.baseline_results
    
    def get_baseline_statistics(self):
        """Get baseline correction statistics."""
        if not self.baseline_results:
            return "No baseline corrections applied."
        return get_baseline_statistics(self.baseline_results)
    
    def print_baseline_summary(self):
        """Print baseline correction summary."""
        if not self.baseline_results:
            print("No baseline corrections applied.")
            return
        print_baseline_summary(self.baseline_results)
    
    def items(self):
        """Return items from all_raman dict."""
        return self.all_raman.items()
    
    def dump_acquired_times_to_file(self, batch_ids, output_file='selected.txt'):
        """Export acquired times to file."""
        return export_acquired_times(self.all_raman, batch_ids, output_file=output_file)
    
    def _print_attribute_info(self):
        """Print available spectrum attributes."""
        if not self.all_raman:
            return
        
        print("\n" + "=" * 70)
        print("AVAILABLE SPECTRUM DATA ATTRIBUTES")
        print("=" * 70)
        
        sample = list(self.all_raman.values())[0]
        
        print("\nFull spectrum data (always available):")
        print("  • x, y → Full wavenumber axis and normalized intensity")
        print("  • raw_y → Original unnormalized intensity values")
        
        if hasattr(sample, 'window_wavenumbers'):
            print("\nWindowed/Reshaped data (created after reshape_range):")
            print("  • window_wavenumbers (alias: p)")
            print("  • window_intensity (alias: q)")
        else:
            print("\nWindowed data (NOT YET CREATED):")
            print("  • Call reshape_all(range) to create:")
            print("    - window_wavenumbers (alias: p)")
            print("    - window_intensity (alias: q)")
        
        if hasattr(sample, 'baseline_corrected_intensity'):
            print("\nBaseline-corrected data (created after baseline correction):")
            print("  • baseline_wavenumbers (alias: u)")
            print("  • baseline_corrected_intensity (alias: v)")
            print("  • current_baseline")
        else:
            print("\nBaseline-corrected data (NOT YET CREATED):")
            print("  • Call baseline_on_data() to create:")
            print("    - baseline_wavenumbers (alias: u)")
            print("    - baseline_corrected_intensity (alias: v)")
            print("    - current_baseline")
        
        print("\nLegacy aliases (p, q, u, v) maintained for backward compatibility.")
        print("=" * 70 + "\n")

    def plot_all_raw(self, alpha=0.6, linewidth=0.8, title=None):
        """Plot all raw spectra in the batch."""
        if title is None:
            title = f"All Raw Spectra ({len(self.all_raman)} samples)"
        plot_all_raw_spectra(self.all_raman, title=title, alpha=alpha, linewidth=linewidth)

    def plot_cleaned(self, baseline_method='poly3', smooth=True, smooth_half_window=3,
                     alpha=0.9, linewidth=0.8, title=None, verbose=False, **kwargs):
        """Perform baseline (poly3 smoothed by default) and plot cleaned spectra."""
        if title is None:
            title = f"Baseline-Corrected Spectra ({len(self.all_raman)} samples)"
        plot_cleaned_spectra(
            self.all_raman,
            baseline_method=baseline_method,
            smooth=smooth,
            smooth_half_window=smooth_half_window,
            title=title,
            alpha=alpha,
            linewidth=linewidth,
            verbose=verbose,
            **kwargs,
        )

    def plot_labelled_svd(
        self,
        label_to_acquired,
        color_map=None,
        n_components=2,
        title='SVD (2D) of Raman Spectra by Label',
        xlabel='SVD Component 1',
        ylabel='SVD Component 2',
        figsize=(8, 6),
        point_size=40,
        alpha=0.8,
        grid=True,
        legend_title='Label',
        legend_outside=True,
        list_of_raman_batch_indexes=False,
        batch_indices=None,
        label_for_indices='selected',
        standardize=False,
        plot_3d=False,
        perform_svm_algorithm=False,
        balance_dataset=True,
        svm_kernel='rbf',
        svm_C=1.0,
        test_size=0.2,
        random_state=42,
    ):
        """Plot a labelled SVD scatter using baseline-corrected intensities (.v) with optional SVM classification.

        Parameters
        ----------
        label_to_acquired : dict[str, list[str]]
            Mapping of label to acquisition timestamps ("DD.MM.YYYY HH:MM:SS").
        color_map : dict[str, str] | None
            Optional mapping of label to color; falls back to a 10-color palette.
        n_components : int
            Number of SVD components to compute (default 2). Only the first two
            components are plotted.
        title, xlabel, ylabel : str
            Text for the plot title and axes labels; adjust to taste.
        figsize : tuple[float, float]
            Matplotlib figure size in inches.
        point_size : float
            Marker size for scatter points.
        alpha : float
            Marker transparency (0-1).
        grid : bool
            Whether to show a background grid.
        legend_title : str
            Title displayed above the legend.
        legend_outside : bool
            Place legend outside the plot on the right when True.
        list_of_raman_batch_indexes : bool
            When True, use `batch_indices` instead of `label_to_acquired`.
        batch_indices : list[int] | dict[str, list[int]] | None
            Either a flat list of indices (single label) or a dict mapping labels to lists.
        label_for_indices : str
            Label name applied when `batch_indices` is a flat list.
        standardize : bool
            Whether to standardize (z-score) data before SVD (default False).
        plot_3d : bool
            Whether to create a 3D plot instead of 2D (default False). Requires n_components >= 3.
        perform_svm_algorithm : bool
            Whether to perform SVM classification after SVD (default False).
        balance_dataset : bool
            Whether to balance the dataset before SVM training (default True).
        svm_kernel : str
            Kernel type for SVM ('linear', 'rbf', 'poly', 'sigmoid'; default 'rbf').
        svm_C : float
            Regularization parameter for SVM (default 1.0).
        test_size : float
            Proportion of dataset to include in the test split (default 0.2).
        random_state : int
            Random state for reproducibility (default 42).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure (or list of figures if SVM is performed).
        df : pandas.DataFrame
            Data frame with components, labels, and acquisition times.
        info : dict
            Diagnostics including missing acquisitions, SVD details, and SVM results if applicable.
        """
        if list_of_raman_batch_indexes:
            if not batch_indices:
                raise ValueError("batch_indices is empty; provide at least one index.")

            missing_index_ids = []
            label_to_acquired = {}

            if isinstance(batch_indices, dict):
                for lbl, idx_list in batch_indices.items():
                    label_to_acquired[lbl] = []
                    for idx in idx_list:
                        spectrum = self.all_raman.get(idx)
                        if spectrum is None:
                            missing_index_ids.append(idx)
                        else:
                            label_to_acquired[lbl].append(spectrum.acquired)
            else:
                label_to_acquired[label_for_indices] = []
                for idx in batch_indices:
                    spectrum = self.all_raman.get(idx)
                    if spectrum is None:
                        missing_index_ids.append(idx)
                    else:
                        label_to_acquired[label_for_indices].append(spectrum.acquired)

            if missing_index_ids:
                print(f"Warning: Missing batch indices: {missing_index_ids}")
        else:
            missing_index_ids = []

        if not label_to_acquired:
            raise ValueError("No labels/acquisitions provided for SVD plot.")
        if not self.all_raman:
            raise ValueError("No spectra loaded in RamanBatch.")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.decomposition import TruncatedSVD

        acquired_lookup = {spec.acquired: spec for spec in self.all_raman.values()}

        X = []
        y = []
        used_acquired = []
        missing = {}
        base_length = None

        for label, acquired_list in label_to_acquired.items():
            missing_for_label = []
            for acquired_time in acquired_list:
                spectrum = acquired_lookup.get(acquired_time)
                if spectrum is None:
                    missing_for_label.append(acquired_time)
                    continue

                if not hasattr(spectrum, 'v'):
                    raise ValueError(
                        f"Spectrum '{acquired_time}' lacks baseline-corrected data (.v). "
                        "Run baseline_on_data() before plotting SVD."
                    )

                vector = np.asarray(spectrum.v)
                if base_length is None:
                    base_length = vector.shape[0]
                elif vector.shape[0] != base_length:
                    raise ValueError(
                        "All spectra must share the same length. "
                        f"'{acquired_time}' has length {vector.shape[0]}, expected {base_length}. "
                        "Reshape/interpolate to a common size first."
                    )

                X.append(vector)
                y.append(label)
                used_acquired.append(acquired_time)

            if missing_for_label:
                missing[label] = missing_for_label

        if not X:
            raise ValueError("No spectra matched the provided acquisition times.")

        data_matrix = np.vstack(X)
        
        # Optional standardization (z-score normalization)
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_matrix = scaler.fit_transform(data_matrix)
        
        svd = TruncatedSVD(n_components=n_components)
        embedding = svd.fit_transform(data_matrix)
        
        # Enforce consistent sign convention: make largest absolute value positive
        # This ensures the same data produces the same orientation every time
        for i in range(embedding.shape[1]):
            max_abs_idx = np.argmax(np.abs(embedding[:, i]))
            if embedding[max_abs_idx, i] < 0:
                embedding[:, i] *= -1
                svd.components_[i, :] *= -1

        unique_labels = sorted(set(y))
        label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
        int_to_label = {v: k for k, v in label_to_int.items()}
        if color_map is None:
            default_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
            color_map = {label: default_colors[i % len(default_colors)]
                         for i, label in enumerate(unique_labels)}

        # =====================================================================
        # SVM Classification (if requested)
        # =====================================================================
        svm_results = None
        svm_figures = []
        
        if perform_svm_algorithm:
            from sklearn.svm import SVC
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (
                confusion_matrix, classification_report, 
                accuracy_score, precision_recall_fscore_support
            )
            from imblearn.over_sampling import RandomOverSampler
            import seaborn as sns
            
            print("\n" + "="*60)
            print("SVM CLASSIFICATION ANALYSIS")
            print("="*60)
            
            # 1. Analyze dataset distribution
            label_counts = {}
            for label in unique_labels:
                label_counts[label] = y.count(label)
            
            print("\nOriginal dataset distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} samples")
            print(f"Total samples: {len(y)}")
            
            # 2. Balance dataset if requested
            X_for_svm = embedding[:, :3] if plot_3d else embedding[:, :2]
            y_for_svm = np.array(y)
            
            if balance_dataset and len(unique_labels) > 1:
                print("\nBalancing dataset using RandomOverSampler...")
                ros = RandomOverSampler(random_state=random_state)
                X_for_svm, y_for_svm = ros.fit_resample(X_for_svm, y_for_svm)
                
                # Show balanced distribution
                balanced_counts = {}
                for label in unique_labels:
                    balanced_counts[label] = list(y_for_svm).count(label)
                
                print("Balanced dataset distribution:")
                for label, count in balanced_counts.items():
                    print(f"  {label}: {count} samples")
                print(f"Total samples after balancing: {len(y_for_svm)}")
            
            # 3. Split data into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_for_svm, y_for_svm, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_for_svm if len(unique_labels) > 1 else None
            )
            
            print(f"\nTrain set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # 4. Train SVM classifier
            print(f"\nTraining SVM with kernel='{svm_kernel}', C={svm_C}...")
            svm_classifier = SVC(kernel=svm_kernel, C=svm_C, random_state=random_state)
            svm_classifier.fit(X_train, y_train)
            
            # 5. Make predictions
            y_pred = svm_classifier.predict(X_test)
            
            # 6. Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            print("\n" + "-"*60)
            print("CLASSIFICATION METRICS")
            print("-"*60)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            # 7. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            
            # Create confusion matrix plot
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels,
                ax=ax_cm, cbar_kws={'label': 'Count'}
            )
            ax_cm.set_title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_xlabel('Predicted Label')
            plt.tight_layout()
            svm_figures.append(fig_cm)
            
            # 8. Plot decision boundaries
            if plot_3d:
                # 3D case: Create projection plots with decision boundaries
                print("\n3D Decision Boundary: Creating 2D projection views...")
                
                fig_db3d = plt.figure(figsize=(15, 5))
                
                # Create meshgrid for decision boundary
                h = 0.02  # step size in the mesh
                x_min, x_max = X_for_svm[:, 0].min() - 1, X_for_svm[:, 0].max() + 1
                y_min, y_max = X_for_svm[:, 1].min() - 1, X_for_svm[:, 1].max() + 1
                z_min, z_max = X_for_svm[:, 2].min() - 1, X_for_svm[:, 2].max() + 1
                
                # Plot 1: Component 1 vs Component 2 (fixing Component 3 at mean)
                ax1 = fig_db3d.add_subplot(131)
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h)
                )
                Z_mean = np.full(xx.ravel().shape, X_for_svm[:, 2].mean())
                Z_labels = svm_classifier.predict(
                    np.c_[xx.ravel(), yy.ravel(), Z_mean]
                )
                Z = np.vectorize(label_to_int.get)(Z_labels).reshape(xx.shape)
                
                ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                for label in unique_labels:
                    idx = [i for i, lbl in enumerate(y_for_svm) if lbl == label]
                    ax1.scatter(
                        X_for_svm[idx, 0], X_for_svm[idx, 1],
                        label=label, s=30, alpha=0.6,
                        color=color_map.get(label, '#7f7f7f'),
                        edgecolors='black', linewidths=0.5
                    )
                ax1.set_xlabel('SVD Component 1')
                ax1.set_ylabel('SVD Component 2')
                ax1.set_title('View: Comp1 vs Comp2')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Component 1 vs Component 3 (fixing Component 2 at mean)
                ax2 = fig_db3d.add_subplot(132)
                xx, zz = np.meshgrid(
                    np.arange(x_min, x_max, h),
                    np.arange(z_min, z_max, h)
                )
                Y_mean = np.full(xx.ravel().shape, X_for_svm[:, 1].mean())
                Z_labels = svm_classifier.predict(
                    np.c_[xx.ravel(), Y_mean, zz.ravel()]
                )
                Z = np.vectorize(label_to_int.get)(Z_labels).reshape(xx.shape)
                
                ax2.contourf(xx, zz, Z, alpha=0.3, cmap='viridis')
                for label in unique_labels:
                    idx = [i for i, lbl in enumerate(y_for_svm) if lbl == label]
                    ax2.scatter(
                        X_for_svm[idx, 0], X_for_svm[idx, 2],
                        label=label, s=30, alpha=0.6,
                        color=color_map.get(label, '#7f7f7f'),
                        edgecolors='black', linewidths=0.5
                    )
                ax2.set_xlabel('SVD Component 1')
                ax2.set_ylabel('SVD Component 3')
                ax2.set_title('View: Comp1 vs Comp3')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Component 2 vs Component 3 (fixing Component 1 at mean)
                ax3 = fig_db3d.add_subplot(133)
                yy, zz = np.meshgrid(
                    np.arange(y_min, y_max, h),
                    np.arange(z_min, z_max, h)
                )
                X_mean = np.full(yy.ravel().shape, X_for_svm[:, 0].mean())
                Z_labels = svm_classifier.predict(
                    np.c_[X_mean, yy.ravel(), zz.ravel()]
                )
                Z = np.vectorize(label_to_int.get)(Z_labels).reshape(yy.shape)
                
                ax3.contourf(yy, zz, Z, alpha=0.3, cmap='viridis')
                for label in unique_labels:
                    idx = [i for i, lbl in enumerate(y_for_svm) if lbl == label]
                    ax3.scatter(
                        X_for_svm[idx, 1], X_for_svm[idx, 2],
                        label=label, s=30, alpha=0.6,
                        color=color_map.get(label, '#7f7f7f'),
                        edgecolors='black', linewidths=0.5
                    )
                ax3.set_xlabel('SVD Component 2')
                ax3.set_ylabel('SVD Component 3')
                ax3.set_title('View: Comp2 vs Comp3')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                
                fig_db3d.suptitle(
                    f'3D SVM Decision Boundaries - 2D Projections (Accuracy: {accuracy:.4f})',
                    fontsize=14, fontweight='bold'
                )
                plt.tight_layout()
                svm_figures.append(fig_db3d)
                
            else:
                # 2D case: Plot decision boundary directly
                print("\n2D Decision Boundary: Plotting...")
                
                fig_db, ax_db = plt.subplots(figsize=(10, 8))
                
                # Create meshgrid
                h = 0.02  # step size in the mesh
                x_min, x_max = X_for_svm[:, 0].min() - 1, X_for_svm[:, 0].max() + 1
                y_min, y_max = X_for_svm[:, 1].min() - 1, X_for_svm[:, 1].max() + 1
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h)
                )
                
                # Predict on meshgrid
                Z_labels = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = np.vectorize(label_to_int.get)(Z_labels).reshape(xx.shape)
                
                # Plot decision boundary
                ax_db.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                
                # Plot data points
                for label in unique_labels:
                    idx = [i for i, lbl in enumerate(y_for_svm) if lbl == label]
                    ax_db.scatter(
                        X_for_svm[idx, 0], X_for_svm[idx, 1],
                        label=label, s=50, alpha=0.7,
                        color=color_map.get(label, '#7f7f7f'),
                        edgecolors='black', linewidths=1
                    )
                
                # Mark support vectors if available
                if hasattr(svm_classifier, 'support_vectors_'):
                    ax_db.scatter(
                        svm_classifier.support_vectors_[:, 0],
                        svm_classifier.support_vectors_[:, 1],
                        s=100, facecolors='none', edgecolors='red',
                        linewidths=2, label='Support Vectors'
                    )
                
                ax_db.set_xlabel('SVD Component 1')
                ax_db.set_ylabel('SVD Component 2')
                ax_db.set_title(
                    f'SVM Decision Boundary (Accuracy: {accuracy:.4f})\n'
                    f'Kernel: {svm_kernel}, C: {svm_C}',
                    fontweight='bold'
                )
                ax_db.legend()
                ax_db.grid(True, alpha=0.3)
                plt.tight_layout()
                svm_figures.append(fig_db)
            
            # Store SVM results
            svm_results = {
                'classifier': svm_classifier,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': classification_report(y_test, y_pred, zero_division=0),
                'balanced_data': (X_for_svm, y_for_svm) if balance_dataset else None,
            }
            
            print("\n" + "="*60)
            print("SVM CLASSIFICATION COMPLETED")
            print("="*60 + "\n")
        
        # =====================================================================
        # Original SVD Plot
        # =====================================================================
        if plot_3d:
            if n_components < 3:
                raise ValueError("plot_3d=True requires n_components >= 3")
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            for label in unique_labels:
                idx = [i for i, lbl in enumerate(y) if lbl == label]
                ax.scatter(
                    embedding[idx, 0],
                    embedding[idx, 1],
                    embedding[idx, 2],
                    label=label,
                    s=point_size,
                    alpha=alpha,
                    color=color_map.get(label, '#7f7f7f'),
                    edgecolors='white',
                    linewidths=0.5,
                )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('SVD Component 3')
            if grid:
                ax.grid(True, alpha=0.3)
            if legend_outside:
                ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1.15, 0.5))
            else:
                ax.legend(title=legend_title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for label in unique_labels:
                idx = [i for i, lbl in enumerate(y) if lbl == label]
                ax.scatter(
                    embedding[idx, 0],
                    embedding[idx, 1] if n_components > 1 else np.zeros(len(idx)),
                    label=label,
                    s=point_size,
                    alpha=alpha,
                    color=color_map.get(label, '#7f7f7f'),
                    edgecolors='white',
                    linewidths=0.5,
                )

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if grid:
                ax.grid(True, alpha=0.3)

            if legend_outside:
                fig.subplots_adjust(right=0.78)
                ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1.02, 0.5))
            else:
                ax.legend(title=legend_title)

        df = pd.DataFrame({
            'label': y,
            'acquired': used_acquired,
            'component_1': embedding[:, 0],
            'component_2': embedding[:, 1] if n_components > 1 else np.zeros(len(y)),
        })

        info = {
            'missing_acquired': missing,
            'missing_batch_indices': missing_index_ids,
            'singular_values': svd.singular_values_,
            'components': svd.components_,
            'model': svd,
            'embedding': embedding,
            'data_frame': df,
        }
        
        # Add SVM results to info if performed
        if perform_svm_algorithm:
            info['svm_results'] = svm_results
            # Return list of figures: [main_plot, confusion_matrix, decision_boundary]
            all_figures = [fig] + svm_figures
            return all_figures, df, info
        
        return fig, df, info

    def plot_labelled_svd_3d_animation(
        self,
        label_to_acquired,
        color_map=None,
        n_components=3,
        title='SVD (3D) of Raman Spectra by Label',
        xlabel='SVD Component 1',
        ylabel='SVD Component 2',
        figsize=(8, 6),
        point_size=40,
        alpha=0.8,
        grid=True,
        legend_title='Label',
        list_of_raman_batch_indexes=False,
        batch_indices=None,
        label_for_indices='selected',
        standardize=False,
        rotation_speed=2,
        elevation=20,
        num_frames=90,
        interval=50,
        save_path=None,
    ):
        """Create an animated 3D rotation of labelled SVD scatter.

        Parameters
        ----------
        label_to_acquired : dict[str, list[str]]
            Mapping of label to acquisition timestamps ("DD.MM.YYYY HH:MM:SS").
        color_map : dict[str, str] | None
            Optional mapping of label to color; falls back to a 10-color palette.
        n_components : int
            Number of SVD components to compute (default 3, minimum 3 for 3D).
        title, xlabel, ylabel : str
            Text for the plot title and axes labels.
        figsize : tuple[float, float]
            Matplotlib figure size in inches.
        point_size : float
            Marker size for scatter points.
        alpha : float
            Marker transparency (0-1).
        grid : bool
            Whether to show a background grid.
        legend_title : str
            Title displayed above the legend.
        list_of_raman_batch_indexes : bool
            When True, use `batch_indices` instead of `label_to_acquired`.
        batch_indices : list[int] | dict[str, list[int]] | None
            Either a flat list of indices (single label) or a dict mapping labels to lists.
        label_for_indices : str
            Label name applied when `batch_indices` is a flat list.
        standardize : bool
            Whether to standardize (z-score) data before SVD (default False).
        rotation_speed : float
            Degrees per frame of azimuthal rotation (default 2).
        elevation : float
            Elevation angle for the camera view (default 20 degrees).
        num_frames : int
            Total number of frames in the animation (default 90).
        interval : int
            Delay between frames in milliseconds (default 50).
        save_path : str | None
            If provided, save animation to this path (e.g., 'animation.gif' or 'animation.mp4').

        Returns
        -------
        animation : matplotlib.animation.FuncAnimation
            The animation object. Call plt.show() to display or use save_path to save.
        df : pandas.DataFrame
            Data frame with components, labels, and acquisition times.
        info : dict
            Diagnostics including missing acquisitions and SVD details.
        """
        if n_components < 3:
            raise ValueError("3D animation requires n_components >= 3")

        if list_of_raman_batch_indexes:
            if not batch_indices:
                raise ValueError("batch_indices is empty; provide at least one index.")

            missing_index_ids = []
            label_to_acquired = {}

            if isinstance(batch_indices, dict):
                for lbl, idx_list in batch_indices.items():
                    label_to_acquired[lbl] = []
                    for idx in idx_list:
                        spectrum = self.all_raman.get(idx)
                        if spectrum is None:
                            missing_index_ids.append(idx)
                        else:
                            label_to_acquired[lbl].append(spectrum.acquired)
            else:
                label_to_acquired[label_for_indices] = []
                for idx in batch_indices:
                    spectrum = self.all_raman.get(idx)
                    if spectrum is None:
                        missing_index_ids.append(idx)
                    else:
                        label_to_acquired[label_for_indices].append(spectrum.acquired)

            if missing_index_ids:
                print(f"Warning: Missing batch indices: {missing_index_ids}")
        else:
            missing_index_ids = []

        if not label_to_acquired:
            raise ValueError("No labels/acquisitions provided for SVD plot.")
        if not self.all_raman:
            raise ValueError("No spectra loaded in RamanBatch.")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.decomposition import TruncatedSVD

        acquired_lookup = {spec.acquired: spec for spec in self.all_raman.values()}

        X = []
        y = []
        used_acquired = []
        missing = {}
        base_length = None

        for label, acquired_list in label_to_acquired.items():
            missing_for_label = []
            for acquired_time in acquired_list:
                spectrum = acquired_lookup.get(acquired_time)
                if spectrum is None:
                    missing_for_label.append(acquired_time)
                    continue

                if not hasattr(spectrum, 'v'):
                    raise ValueError(
                        f"Spectrum '{acquired_time}' lacks baseline-corrected data (.v). "
                        "Run baseline_on_data() before plotting SVD."
                    )

                vector = np.asarray(spectrum.v)
                if base_length is None:
                    base_length = vector.shape[0]
                elif vector.shape[0] != base_length:
                    raise ValueError(
                        "All spectra must share the same length. "
                        f"'{acquired_time}' has length {vector.shape[0]}, expected {base_length}. "
                        "Reshape/interpolate to a common size first."
                    )

                X.append(vector)
                y.append(label)
                used_acquired.append(acquired_time)

            if missing_for_label:
                missing[label] = missing_for_label

        if not X:
            raise ValueError("No spectra matched the provided acquisition times.")

        data_matrix = np.vstack(X)
        
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_matrix = scaler.fit_transform(data_matrix)
        
        svd = TruncatedSVD(n_components=n_components)
        embedding = svd.fit_transform(data_matrix)
        
        # Enforce consistent sign convention
        for i in range(embedding.shape[1]):
            max_abs_idx = np.argmax(np.abs(embedding[:, i]))
            if embedding[max_abs_idx, i] < 0:
                embedding[:, i] *= -1
                svd.components_[i, :] *= -1

        unique_labels = sorted(set(y))
        if color_map is None:
            default_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
            color_map = {label: default_colors[i % len(default_colors)]
                         for i, label in enumerate(unique_labels)}

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        scatter_plots = []
        for label in unique_labels:
            idx = [i for i, lbl in enumerate(y) if lbl == label]
            sc = ax.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                embedding[idx, 2],
                label=label,
                s=point_size,
                alpha=alpha,
                color=color_map.get(label, '#7f7f7f'),
                edgecolors='white',
                linewidths=0.5,
            )
            scatter_plots.append(sc)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel('SVD Component 3')
        if grid:
            ax.grid(True, alpha=0.3)
        ax.legend(title=legend_title, loc='upper left')

        def update(frame):
            ax.view_init(elev=elevation, azim=frame * rotation_speed)
            return scatter_plots

        anim = FuncAnimation(
            fig, update, frames=num_frames, interval=interval, blit=False
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
            else:
                anim.save(save_path)
            print(f"Animation saved to {save_path}")

        df = pd.DataFrame({
            'label': y,
            'acquired': used_acquired,
            'component_1': embedding[:, 0],
            'component_2': embedding[:, 1],
            'component_3': embedding[:, 2] if n_components > 2 else np.zeros(len(y)),
        })

        info = {
            'missing_acquired': missing,
            'missing_batch_indices': missing_index_ids,
            'singular_values': svd.singular_values_,
            'components': svd.components_,
            'model': svd,
            'embedding': embedding,
            'data_frame': df,
        }

        return anim, df, info