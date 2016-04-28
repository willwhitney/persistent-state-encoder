import os
import sys

dry_run = '--dry-run' in sys.argv
local   = '--local' in sys.argv
detach  = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


networks_prefix = "networks"

base_networks = {
    }


# Don't give it a save name - that gets generated for you
# jobs = [
#         {
#             "import": "onestep",
#         },
#
#
#     ]

jobs = []

learning_rate_options = [2e-4]
max_heads_options = [20]
cost_decay_rate_options = [0.9, 0.95]
cost_per_head_options = [0.1, 0.01]

for learning_rate in learning_rate_options:
    for max_heads in max_heads_options:
        for cost_decay_rate in cost_decay_rate_options:
            for cost_per_head in cost_per_head_options:
                job = {
                        "learning_rate": learning_rate,
                        "max_heads": max_heads,
                        "cost_decay_rate": cost_decay_rate,
                        "cost_per_head": cost_per_head,

                        "gpu": True,
                    }
                jobs.append(job)


if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    jobname = "variable"
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " --" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                flagstring = flagstring + " --" + flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " --" + flag + " " + networks_prefix + "/" + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = "th variable_main.lua" + flagstring

    print(jobcommand)
    if local and not dry_run:
        if detach:
            os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
        else:
            os.system(jobcommand)

    else:
        with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write(jobcommand)

        if not dry_run:
            if 'gpu' in job and job['gpu']:
                os.system("sbatch -N 1 -c 2 --gres=gpu:titan-x:1 --mem=8000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
            else:
                os.system("sbatch -N 1 -c 2 --mem=8000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
