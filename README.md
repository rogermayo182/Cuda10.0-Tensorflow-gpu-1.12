# Cuda10.0-Tensorflow-gpu-1.12
Install CUDA 10.0 Tensorflow-1.12 on ubuntu 18.04

At the present time,the latest tensorflow-gpu-1.12 version installed by system pip is not compatiable to CUDA 10.0,for it was build by CUDA 9.0,so if you want to use the latest version tensorflow-gpu with CUDA 10.0 in 18.04,you need to build from source.This is going to be a tutorial on how to install tensorflow 1.12 GPU version. We will also be installing CUDA 10.0 and cuDNN 7.3.1 along with the GPU version of tensorflow 1.12.



Step 1: Update and Upgrade your system:



#suggest to change the apt source to local sites



sudo apt-get update



sudo apt-get upgrade



Step 2: Verify You Have a CUDA-Capable GPU:



lspci | grep -i nvidia



Note GPU model. eg. Quadro K620



If you do not see any settings, update the PCI hardware database that Linux maintains by entering update-pciids (generally found in /sbin) at the command line and rerun the previous lspci command.



If your graphics card is from NVIDIA then goto CUDA GPUs and verify if listed in CUDA enabled gpu list.



Note down its Compute Capability. eg. GeForce 840M 5.0



Step 3: Verify You Have a Supported Version of Linux:



To determine which distribution and release number you’re running, type the following at the command line:



uname -m && cat /etc/*release



The x86_64 line indicates you are running on a 64-bit system which is supported by cuda 10



Step 4: Install Dependencies:



Required to compile from source:



sudo apt-get install build-essential



sudo apt-get install cmake git unzip zip



sudo apt-get install python-dev python-pip



#suggest to change pip source site

#mkdir ~/.pip

#gedit ~/.pip/pip.conf

#modify pip.conf content：

#[global]

#index-url = Simple Index



Step 5: Install linux kernel header:



Goto terminal and type:



uname -r



You can get like “4.15.0-36-generic”. Note down linux kernel version.



To install linux header supported by your linux kernel do following:



sudo apt-get install linux-headers-$(uname -r)



Step 6: Install NVIDIA CUDA 10.0:

If you have been installed any version of CUDA,remove previous cuda installation use follow action:

sudo apt-get purge nvidia*



sudo apt-get autoremove



sudo apt-get autoclean



sudo rm -rf /usr/local/cuda*



Install cuda For Ubuntu 18.04 :



sudo apt install dracut-core



sudo gedit /etc/modprobe.d/blacklist.conf



append one line to the file:

blacklist nouveau



sudo update-initramfs -u



reboot



Download CUDA install files,download address:CUDA Toolkit 10.0 Download.



excute the installation:



sudo ./cuda_10.0.130_410.48_linux.run



Step 7: Reboot the system to load the NVIDIA drivers.



Step 8: Go to terminal and type:



echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc



echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc



source ~/.bashrc



sudo ldconfig



nvidia-smi



Check driver version probably Driver Version: 410.48



(not likely) If you got nvidia-smi is not found then you have unsupported linux kernel installed. Comment your linux kernel version noted in step 5.



You can check your cuda installation using following sample:



cd ~/NVIDIA_CUDA-10.0_Samples/1_Utilities/deviceQuery



make



./deviceQuery



the result will show the information about you GPU devices



Step 9: Install cuDNN 7.3.1:



NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.



Goto NVIDIA cuDNN and download Login and agreement required,You can download the file without login in the follow address:

https://pan.baidu.com/s/1MnUc03WLTNPni2UhMj_pnw



Download the following:



cuDNN v7.3.1 Library for Linux [ cuda 10.0]



After downloaded folder and in terminal perform following:



tar -xf cudnn-10.0-linux-x64-v7.3.1.20.tgz



sudo cp -R cuda/include/* /usr/local/cuda-10.0/include



sudo cp -R cuda/lib64/* /usr/local/cuda-10.0/lib64



Step 10: Install NCCL 2.3.5:



NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives that are performance optimized for NVIDIA GPUs



Go and attend survey to https://developer.nvidia.com/nccl/nccl-download to download Nvidia NCCL.You can download the file without login in the follow address:

https://pan.baidu.com/s/1y1X_zxU156K-zyKRC7GZpQ



Download following:



Download NCCL v2.3.5, for CUDA 10.0 -> NCCL 2.3.5 O/S agnostic and CUDA 10.0



Goto downloaded folder and in terminal perform following:



tar -xf nccl_2.3.5-2+cuda10.0_x86_64.txz



cd nccl_2.3.5-2+cuda10.0_x86_64



sudo cp -R * /usr/local/cuda-10.0/targets/x86_64-linux/



sudo ldconfig



Step 11: Install Dependencies



Use following if not in active virtual environment.



pip install -U --user pip six numpy wheel mock



pip install -U --user keras_applications==1.0.5 --no-deps



pip install -U --user keras_preprocessing==1.0.3 --no-deps



Step 12: Configure Tensorflow from source:



Download bazel:



cd ~/



wget https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh



chmod +x bazel-0.18.1-installer-linux-x86_64.sh



./bazel-0.18.1-installer-linux-x86_64.sh --user



echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc



Reload environment variables



source ~/.bashrc

sudo ldconfig



Start the process of building TensorFlow by downloading latest tensorflow 1.12 .



cd ~/



git clone tensorflow/tensorflow



cd tensorflow



git checkout r1.12



./configure



Give python path in



Please specify the location of python. [Default is /usr/bin/python]



Press enter two times



Do you wish to build TensorFlow with Apache Ignite support? [Y/n]: Y



Do you wish to build TensorFlow with XLA JIT support? [Y/n]: Y



Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N



Do you wish to build TensorFlow with ROCm support? [y/N]: N



Do you wish to build TensorFlow with CUDA support? [y/N]: Y



Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.0



Please specify the location where CUDA 10.0 toolkit is installed. Refer to Home for more details. [Default is /usr/local/cuda]: /usr/local/cuda-10.0



Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7.3.1



Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda-10.0/



Do you wish to build TensorFlow with TensorRT support? [y/N]: N



Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 2.3.5



Please specify the location where NCCL 2.3.5 is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda-10.0/targets/x86_64-linux/



Now we need compute capability which we have noted at step 1 eg. 5.0



Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 5.0] 5.0



Do you want to use clang as CUDA compiler? [y/N]: N



Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc



Do you wish to build TensorFlow with MPI support? [y/N]: N



Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=native



Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:N



Configuration finished



Step 13: Build Tensorflow using bazel



The next step in the process to install tensorflow GPU version will be to build tensorflow using bazel. This process takes a fairly long time.



To build a pip package for TensorFlow you would typically invoke the following command:



bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package



Note:-



add "--config=mkl" if you want Intel MKL support for newer intel cpu for faster training on cpu



add "--config=monolithic" if you want static monolithic build (try this if build failed)



add "--local_resources 2048,.5,1.0" if your PC has low ram causing Segmentation fault or other related errors



This process will take a lot of time. It may take 3- 4 hours or maybe even more.



Also if you got error like Segmentation Fault then try again it usually worked.



The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the tensorflow_pkg directory:



To build whl file issue following command:



bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg



To install tensorflow with pip:



cd tensorflow_pkg



sudo pip install tensorflow*.whl



Note : if you got error like unsupported platform then make sure you are running correct pip command associated with the python you used while configuring tensorflow build.



Step 14: Verify Tensorflow installation



Run in terminal



python



import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))



The system outputs Tensorflow load information and the 'Hello,TensorFlow!'



Success! You have now successfully installed tensorflow-gpu 1.12 on your machine.



Reference:https://www.python36.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/2/
