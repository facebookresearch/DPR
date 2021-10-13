# COIL Retriever
We provide several implementaions of COIL retriever,
- A fast batched retriver that uses Cython extension, `retriever-fast.py`. This retriever is substantially faster (10x on our system). Use it whenever you can if performance is a concern.
- A pure python batched retriver that is based on external packages, `retriever-cb.py`.
- A pure python sequential retriever, `retriever-compat.py`. Practical use of it is deprecated in favor of the other two.
It remains in the repo as the most expressive implementation for educational purpose.

## Fast Retriver
It has come to my attention that `pytorch_scatter` does not scale well to multiple cores. I finally decided to write a C binding. While a pure C/C++ implementatoin 
is typically the best for realworld setups, I hope this hybrid implementation can offer a sense of how much C code can speed up the stack.

To run the fast retriver, first compile the Cython extension. You will need cython and in addtion a c++ compiler for this.
```
cd retriver/retriever_ext
pip install Cython
python setup.py build_ext --inplace
```
In extreme cases where you cannot get access to a compiler, consider use the pure python batched retirever.
## Running Retrieval
To do retrieval, run the following steps,

(Note that there is no dependency in the for loop within each step, meaning that if you are on a cluster, you can distribute the jobs across nodes using `srun` or `qsub`.)

1) build document index shards (pick number of shards based on your setup, we use 10 here)
```
for i in $(seq 0 9)  
do  
 python retriever/sharding.py \  
   --n_shards 10 \  
   --shard_id $i \  
   --dir $ENCODE_OUT_DIR \  
   --save_to $INDEX_DIR \  
   --use_torch
done  
```
2) reformat encoded query
```
python retriever/format_query.py \  
  --dir $ENCODE_QRY_OUT_DIR \  
  --save_to $QUERY_DIR \  
  --as_torch
```

3) retrieve from each shard (pick retriever based on your setup)
```
for i in $(seq -f "%02g" 0 9)  
do  
  python retriever/{retriver-fast|retriever-cb|retriever-compat}.py \  
      --query $QUERY_DIR \  
      --doc_shard $INDEX_DIR/shard_${i} \  
      --top 1000 \  
      --save_to ${SCORE_DIR}/intermediate/shard_${i}.pt \
      --batch_size 512  # only retriver-fast, retriever-cb have this argument
done 
```
when using batched retriver `retriver-fast` or `retriever-cb`, set the batch size based on your hardware to get the best performance.

4) merge scores from all shards
```
python retriever/merger.py \  
  --score_dir ${SCORE_DIR}/intermediate/ \  
  --query_lookup  ${QUERY_DIR}/cls_ex_ids.pt \  
  --depth 1000 \  
  --save_ranking_to ${SCORE_DIR}/rank.txt

# format the retrieval result
# e.g. msmarco
python data_helpers/msmarco-passage/score_to_marco.py \  
  --score_file ${SCORE_DIR}/rank.txt
```
