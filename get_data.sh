cd data

git lfs install
git clone https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset
git clone https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark

tar zxvf Cosmos-Reason1-RL-Dataset/robovqa/clips.tar.gz -C Cosmos-Reason1-RL-Dataset/robovqa/
tar zxvf Cosmos-Reason1-Benchmark/robovqa/clips.tar.gz -C Cosmos-Reason1-Benchmark/robovqa/

cd ..
