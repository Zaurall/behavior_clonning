import argparse
import traceback
from pathlib import Path

from tqdm import tqdm

from offline.config import PGTOConfig
from offline.pgto.optimizer import PGTOOptimizer
from offline.segment import get_segment_paths, load_segment
from offline.gdrive import GDriveManager


def main() -> None:
    parser = argparse.ArgumentParser(description="PGTO optimization")
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        required=True,
        help="Total number of workers",
    )
    parser.add_argument(
        "-w",
        "--worker-id",
        type=int,
        required=True,
        help="This worker's ID (0 to num-workers-1)",
    )
    parser.add_argument(
        "-m",
        "--max-segment",
        type=int,
        required=True,
        help="Process segments 0 to max",
    )
    parser.add_argument(
        "--min-segment",
        type=int,
        default=0,
        help="Start processing from this segment ID (default 0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-restart optimization details",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=True,
        help="Torch device to use (cpu, mps, or cuda)",
    )
    parser.add_argument(
        "--service-account-file",
        type=str,
        default="service_key.json",
        help="Path to Google service account JSON key file",
    )
    args = parser.parse_args()

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker-id must be in [0, {args.num_workers - 1}]")

    # Create  config
    config = PGTOConfig(device=args.device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gdrive_manager = None
    if config.use_gdrive:
        gdrive_manager = GDriveManager(config.gdrive_folder_name, service_account_file=args.service_account_file)
        print("Using GDrive for output")

    # Get segments this worker should process
    all_segments = get_segment_paths(Path(config.segments_dir))
    segments = [
        p for p in all_segments if args.min_segment <= int(p.stem) <= args.max_segment
    ]
    
    if gdrive_manager:
        gdrive_manager.sync_file_cache()
        remaining = [
            p for p in segments if not gdrive_manager.file_exists(f"{p.stem}.pt")
        ]
    else:
        remaining = [
            p for p in segments if not (output_dir / f"{p.stem}.pt").exists()
        ]  # Filter first
        
    my_segments = remaining[args.worker_id :: args.num_workers]

    print(f"Worker {args.worker_id}/{args.num_workers}")
    print(f"  Total incomplete: {len(remaining)}")
    print(f"  This worker's share: {len(my_segments)}")
    print(f"  Config: K={config.K}, restarts={config.num_restarts}")

    if not remaining:
        print("Nothing to do!")
        return

    # Initialize optimizer
    optimizer = PGTOOptimizer(config)

    # Process segments
    pbar = tqdm(my_segments, desc=f"Worker {args.worker_id}")
    for segment_path in pbar:
        output_path = output_dir / f"{segment_path.stem}.pt"

        # Double-check not done
        if gdrive_manager:
            if gdrive_manager.file_exists(f"{segment_path.stem}.pt"):
                continue
        elif output_path.exists():
            continue

        try:
            segment = load_segment(segment_path, config=config)
            result = optimizer.optimize(segment, verbose=args.verbose)
            
            result.save(output_path)
            
            if gdrive_manager:
                gdrive_manager.upload_file(output_path)
                # optionally remove local file if you don't want to keep it
                # output_path.unlink() 
            
            pbar.set_postfix(
                {"seg": segment_path.stem, "cost": f"{result.best_cost:.1f}"}
            )

        except Exception as e:
            print(f"\nError on {segment_path.stem}: {e}")
            traceback.print_exc()
            continue

    print(f"Worker {args.worker_id} done!")


if __name__ == "__main__":
    main()
