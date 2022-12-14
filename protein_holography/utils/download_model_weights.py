""" Script to download model weights """

import os
import urllib.request

def download_weights(
    download_dir = "../model_weights",
    name = "best_network.tar.gz"
):

    download_path = os.path.join(download_dir,name)
    try:
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/38488937",
            download_path
            #name
        )
        print("Model weights successfully downloaded to "\
              f"{download_path}")
    except Exception as e:
        print(e)
        github = "https://github.com/StatPhysBio/protein_holography"
        print("Model weights could not be downloaded\n" \
              f"Please see {github} for help")

    os.system(
        f"tar -xvzf {download_path} " \
        "-C ../model_weights"      
    )
    os.system(f"rm {download_path}")

def main():
    download_weights()

if __name__ == "__main__":
    main()
