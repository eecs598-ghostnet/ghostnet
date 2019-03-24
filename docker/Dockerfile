#FROM tensorflow/tensorflow:1.10.0-gpu
FROM pytorch/pytorch




RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    	vim \
    	make autoconf automake libtool pkg-config \
    	gcc \ 
    	libsonic-dev \ 
    	ruby-ronn \ 
    	ruby-kramdown \
    	python-pip \
    	sudo 


RUN pip install torchtext

RUN pip install -U gensim

RUN git clone https://github.com/espeak-ng/espeak-ng

RUN cd espeak-ng \ 
     ./autogen.sh \ 
     ./configure --prefix=/usr \
     make \
     #make -B src/espeak-ng src/speak-ng \ 
     ESPEAK_DATA_PATH=/root/home/espeak-ng/espeak-ng-data LD_LIBRARY_PATH=src:${LD_LIBRARY_PATH} src/espeak-ng ... \
     sudo make LIBDIR=/usr/lib/x86_64-linux-gnu install


