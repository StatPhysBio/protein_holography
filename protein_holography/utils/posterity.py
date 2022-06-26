#
# posterity.py -- Michael Pun
#
# File for recording reproducibility metadata
#

import random
import sys
import git
import time

def get_metadata():
    metadata = {}
    # posterity info

    # command line arguments
    metadata['command_line'] = ' '.join(sys.argv)

    # git hash
    repo = git.Repo(search_parent_directories=True)
    metadata['git_hash'] = repo.head.object.hexsha

    # time ran
    metadata['time'] = time.ctime()
    
    # random seed
    metadata['seed'] = random.randrange(sys.maxsize)

    return metadata

def record_metadata(metadata,dset):
    for k in metadata.keys():
        dset.attrs[k] = metadata[k]

