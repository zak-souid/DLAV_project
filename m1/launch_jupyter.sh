#!/bin/bash -l
#SBATCH --job-name=ipython-trial
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output jupyter-log-%J.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos gpu_free
#SBATCH --reservation=courses
#SBATCH --account=civil-459
 
module load gcc python 
 
source opt/venv-gcc/bin/activate

ipnport=$(shuf -i8000-9999 -n1)
 
jupyter-notebook --no-browser --port=${ipnport} --ip=$(hostname -i)

