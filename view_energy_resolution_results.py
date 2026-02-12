import argparse
import os

import energy_resolution_viewer


def main():
    parser = argparse.ArgumentParser(
        description='Standalone viewer for saved SIMION energy-resolution results.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to summary pickle/csv. If omitted, loads latest file in --results-dir.'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results_v5.2',
        help='Directory containing saved result files.'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Only print summary stats and exit.'
    )
    args = parser.parse_args()

    target_file = args.input
    if not target_file:
        target_file = energy_resolution_viewer.find_latest_summary_file(args.results_dir)
        if not target_file:
            raise FileNotFoundError(
                f"No summary files found in '{args.results_dir}'."
            )

    summary_data = energy_resolution_viewer.load_summary_from_file(target_file)
    valid_points, total_points = energy_resolution_viewer.count_valid_energy_points(summary_data)

    print("=" * 70)
    print("Energy Resolution Result Viewer")
    print("=" * 70)
    print(f"Source file: {os.path.abspath(target_file)}")
    print(f"Field gradients: {len(summary_data)}")
    print(f"Valid points: {valid_points}/{total_points}")

    if args.no_gui:
        return

    launched = energy_resolution_viewer.launch_field_gradient_gui(
        summary_data,
        title_prefix=f'Result Viewer: {os.path.basename(target_file)}'
    )
    if not launched:
        raise RuntimeError("Viewer could not find plottable data.")


if __name__ == '__main__':
    main()
