git lfs install
git clone https://huggingface.co/datasets/qian/III-Retriever
git clone https://huggingface.co/Shitao/RetroMAE_MSMARCO_distill
git clone https://huggingface.co/Shitao/RetroMAE_MSMARCO_finetune
mkdir -p data
mv III-Retriever/* data/ 
mv RetroMAE_MSMARCO_distill/ data/
rm -rf III-Retriever
