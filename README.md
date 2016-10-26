# Deep Learning on GPU cluster in Pasteur

## GPU nodes on the new cluster
Tars is the new cluster in Pasteur, 3 GPUs nodes are available from now on on tars (in the common partition).
Slurm(http://slurm.schedmd.com/) are used as the schedule system.

## Slurm commands for GPU nodes
You can start to test them using:
* the QoS named gpu (time limit of 7 days for a job)
* the gres option to indicate what you want.

There are 1 node with 8 teslaK80 and 2 nodes with 4 teslaM40 each.
If you want to submit on a specific type of GPU, indicate it in the gres option like this: `--gres=gpu[:type]:nb_of_gpu`

## Getting start
### Connect to tars.pasteur.fr
Open a terminal, run:
```bash
ssh USER_NAME@tars.pasteur.fr
```
Now, you are on a intermediate node named submission node.

Notice that, you shouldn't run any computational command on this node, it's only for sending command to computation node.

### Load modules for deep learning (with tensorflow)
```bash
module load cuda/7.5.18 cudnn/v4 test/tensorflow/0.9.0 Python/2.7.11
```
### Install Keras for Deep Learning
On the cluster, you won't have root permission, so you need to use `pip` with `--user`.
```bash
# install keras into user home
# do not use sudo here
pip install keras --user --upgrade
```
### Test installation on submission mode
As mentioned, do not run actuall code on submission node, but you can run a shell to test your installation:
```
KERAS_BACKEND=tensorflow ipython
```
If you type:
```python
import keras
```
You should see:
```
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so.7.5 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so.7.5 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so.4 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so.7.5 locally
```
This means, you are ready to go.

Again, within this shell, you shouldn't run any actuall code, because it's on the submission node, so it doesn't even have a GPU.

In order to run it on the GPU nodes, follow the next step.

### Start an interactive session
```bash
salloc --qos=gpu --gres=gpu:1
```
If success, you will get a shell on the requested node, you will be able to run:
```bash
nvidia-smi
```
to check the gpus you have on the node your are running.
Then, you should be able to run
```
python -c 'import tensorflow'
```
or start a python/ipython session with tensorflow enabled
```
ipython
```

### Run scripts (batch mode)
After trying your code within interactive session, you can wrap your code into a python file, and then run it with `sbatch` or `srun` command:
```bash
sbatch  --qos=gpu --gres=gpu:1  ./your_script
srun    --qos=gpu --gres=gpu:1  ./your_command
```
If you want a specific GPU type, specify it:
```
sbatch  --qos=gpu --gres=gpu:teslaK80:1  ./test_cuda.sh
```
# Question
If you encountered any problem, you may get help in the [technical channel on slack](https://deeplearningclub.slack.com/messages/technical).

If you haven't signup, [signup here](https://deeplearningclub.slack.com/signup).


