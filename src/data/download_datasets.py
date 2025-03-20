import os
import sys
import argparse
import requests
import pandas as pd
from tqdm import tqdm


def download_file(url, destination):
    """
    download a file from url to destination with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def download_movielens(output_dir, dataset="ml-1m"):
    """
    download and prepare movielens dataset

    args:
        output_dir: directory to save the dataset
        dataset: which movielens dataset to download ('ml-1m' or 'ml-20m')

    returns:
        success: whether the download was successful
    """
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # determine url based on dataset
    if dataset == "ml-1m":
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(output_dir, "ml-1m.zip")
        extract_dir = os.path.join(output_dir, "ml-1m")
        csv_path = os.path.join(output_dir, "ml-1m.csv")
    elif dataset == "ml-20m":
        url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
        zip_path = os.path.join(output_dir, "ml-20m.zip")
        extract_dir = os.path.join(output_dir, "ml-20m")
        csv_path = os.path.join(output_dir, "ml-20m.csv")
    else:
        print(f"error: unknown dataset {dataset}")
        return False

    # download dataset if it doesn't exist
    if not os.path.exists(csv_path):
        print(f"downloading {dataset} dataset...")
        download_file(url, zip_path)

        # extract dataset
        print(f"extracting {dataset} dataset...")
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        # convert to csv
        print(f"converting {dataset} dataset to csv...")
        if dataset == "ml-1m":
            ratings_path = os.path.join(extract_dir, "ratings.dat")
            with open(ratings_path, "r", encoding="iso-8859-1") as f, open(
                csv_path, "w"
            ) as out:
                # write header
                out.write("userId,movieId,rating,timestamp\n")

                # write data
                for line in f:
                    user_id, movie_id, rating, timestamp = line.strip().split("::")
                    out.write(f"{user_id},{movie_id},{rating},{timestamp}\n")
        elif dataset == "ml-20m":
            ratings_path = os.path.join(extract_dir, "ratings.csv")
            # just copy the file
            import shutil

            shutil.copyfile(ratings_path, csv_path)

    print(f"{dataset} dataset ready at {csv_path}")
    return True


def main(args=None):
    """Main function"""
    if args is None:
        parser = argparse.ArgumentParser(
            description="Download datasets for recommendation system"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="data",
            help="directory to save the datasets",
        )
        parser.add_argument(
            "--datasets",
            type=str,
            nargs="+",
            default=["ml-1m"],
            choices=["ml-1m", "ml-20m"],
            help="which datasets to download",
        )
        args = parser.parse_args()

    results = {}
    for dataset in args.datasets:
        success = download_movielens(args.output_dir, dataset)
        results[dataset] = success
        if not success:
            print(f"Failed to download {dataset}")

    return results


if __name__ == "__main__":
    main()
