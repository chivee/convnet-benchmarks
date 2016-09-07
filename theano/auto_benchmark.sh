#!/usr/bin/env python

MINIBATCH_SIZE=64
NUMBER_OF_MINIBATCHES=100

# set env var
CUR_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
INSTALL_PATH=~/theano_benchmark


# configure and install
mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
git clone https://github.com/theano/Theano
cd Theano
sudo python setup.py install
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
sudo chown -R $USER: ~/.theano/

echo "Current theano version:"
python -c "import theano; print(theano.version.version)"

cat <<EOF > ~/.theanorc
[global]
device = gpu
floatX = float32
optimizer_including = unsafe

[cuda]
root = /usr/local/cuda-7.5/

[lib]
cnmem = 0.45

[dnn.conv]
algo_fwd = time_once
algo_bwd_filter = time_once
algo_bwd_data = time_once
EOF



# run benchmark
cd $CUR_DIR

run_benchmark () {
    mkdir -p log
    for net in "$@"; do
        cat ~/.theanorc >> log/${net}.log
        python benchmark_imagenet.py -a ${net} -B $MINIBATCH_SIZE -n $NUMBER_OF_MINIBATCHES >> log/${net}.log 2>&1
    done
}

run_benchmark alexnet googlenet vgg


# grep the result
grep 'Forward across\|Forward-Backward' log/*.log  >  log/result
