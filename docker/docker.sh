sudo nvidia-docker run --runtime=nvidia -it --rm --net=host\
  -v `pwd`:/root/home \
  -w /root/home \
  ghostnet:latest \
  /bin/bash


 # -v /media/bucket/droplabV2/ConfidenceMapLearning/VAL:/root/dispnet/VAL \
 # -v /media/bucket/droplabV2/ConfidenceMapLearning/TRAIN:/root/dispnet/TRAIN \
