import os
import sys
from config import Config

throwaway_script = 'batch_script.slurm'
config = Config()

for job, script in config.generate_sbatch_scripts():
    print('writing script')
    f = open(throwaway_script,'w')
    f.write(script)
    f.close()
    print('Submitting job ' + job)
    if os.system('sbatch ' + throwaway_script) != 0:
        raise Exception("Error while submitting a job.")
    print('[ok]')
