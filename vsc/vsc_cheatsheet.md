`qstat -q` -> check the job queue

`module av` -> list of available compiled/optimized software packages (intel or foss toolchain)

`qsub example.pbs` -> submit a job to the queue

`qstat 12345` -> monitor status of job `12345`

`qstat -n 12345` -> show which compute node the job runs on

`showstart 12345` -> show estimated start time of job `12345`

`checkjob 12345` -> check job's resources

`qdel 12345` -> delete the job

`qstat` -> show all submitted jobs

```shell
FIRST_ID=$(qsub job1.sh)
qsub -W depend=afterok:$FIRST_ID job2.sh
```
-> submit jobs that depend on results of others (`afternotok`, `afterany`)

`hostname -f` -> which computer the current script/session/job is on

`du -h` -> human-readable disk usage

`chgrp -R groupname directory` -> add users to the group/project to gain access to the same data

`getent group example` -> list users in the group

`quota` -> print storage quota

```shell
module load worker/version
wsub -batch job.pbs -data data.csv
```
-> spawn a batch of jobs running the job, but with parameters from the data file (./weather -t $temperature -p $pressure -v $volume)

```shell
# Check if the output Directory exists
if [ ! -d "./output" ] ; then
  mkdir ./output
fi
```

`wsub -t 1-100 -batch test_set.pbs` -> run a batch of jobs split over compute resources mentioned in the pbs file (./test_set ${PBS_ARRAYID} -> the actual job + index of the job)

`mam-balance` -> available credits per project

```shell
module load accounting
gquote example.pbs
```
-> get a credit quote for the job

`mam-statement` -> overview of transactions

`mam-list-transactions --summarize` -> summarize transactions in a project

`mam-list-transactions J 12345` -> summarize consumed resources of a completed job

`watch "nvidia-smi` -> monitor GPUs
