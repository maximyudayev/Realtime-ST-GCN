`module av` -> list of available compiled/optimized software packages (intel or foss toolchain)

`module use /apps/leuven/{$VSC_ARCH_LOCAL}/2021a/modules/all` -> use optimized software compiled for the CPU architecture of the node the job is running on

`module load ...` -> load the software into the current environment

`sbatch example.slurm` -> submit a job to the queue (all CLI parameters after the name of the job script are passed as CLI arguments to the job script, hence all sbatch options must be provided before the name of the job script)

`sbatch --array [1-100]%5 ...` -> launch job array with indices 1-100, with at most 5 jobs executing at a time

`slurm_jobinfo <jobid>` -> VSC's custom command to filter and format relevant job information

`scontrol --cluster=<cluster>` -> view Slurm [configuration and state](https://slurm.schedmd.com/scontrol.html)

`scontrol -d show job <jobid>` -> show extensive information about a particular job

`squeue --cluster=<cluster>` -> get information about jobs in the [scheduling queue](https://slurm.schedmd.com/squeue.html)

`sacct --cluster=<cluster>` -> display information about [finished jobs](https://slurm.schedmd.com/sacct.html)

`scancel <jobarrayid>_<jobid> | <jobarrayid>_[jobid-jobid]` -> kills specified job in the job array and range of jobs in the job array, respectively

`sstat --jobs=<job_id_list>` -> displays information about running jobs, specified as a comma-separated list

```shell
--dependency=<type>:jobid:jobid,<type>:jobid:jobid
--dependency=<type>:jobid:jobid?<type>:jobid:jobid
```
-> submit jobs that [depend on results of others](https://docs.vscentrum.be/jobs/job_submission.html#specifying-dependencies) [(`after`, `afterany`, `afterok`, `afternotok`, `singleton`, etc.)](https://slurm.schedmd.com/sbatch.html#OPT_dependency), where `,`, `?` are AND, OR, respectively

```shell
--export=HOME,USER,TERM,PATH=/bin:/sbin,FOO=bar
```
-> propogate environment variables from the session that submitted the job to the job node (recommended to load modules in the jobscripts and not in ~/.bashrc) 

`slurmtop --cluster=<cluster>` -> cluster status overview

`hostname -f` -> which computer the current script/session/job is on

`du -h` -> human-readable disk usage

`chgrp -R groupname directory` -> add users to the group/project to gain access to the same data

`getent group example` -> list users in the group

`myquota` -> print storage quota

<!-- ```shell
module load worker/version
wsub -batch job.pbs -data data.csv
```
-> spawn a batch of jobs running the job, but with parameters from the data file (./weather -t $temperature -p $pressure -v $volume) -->

```shell
# Check if the output Directory exists
if [ ! -d "./output" ] ; then
  mkdir ./output
fi
```

`sam-balance` -> available credits per project

```shell
module load accounting
gquote example.pbs
```
-> get a credit quote for the job

`sam-statement -A <project>` -> overview of transactions

`sam-list-usagerecords -A <project> -s <YYYY-MM-DD> -e <YYYY-MM-DD>` -> summarize transactions in a project

`watch nvidia-smi` -> monitor GPUs
