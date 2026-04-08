#!/bin/bash
# GPU 1의 프로세스(PID 2601843)가 끝나면 LHP-CZSL 학습 시작

TARGET_PID=2601843
echo "Waiting for PID $TARGET_PID (AdaptDPC v2 on GPU 1) to finish..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 30
done

echo "PID $TARGET_PID finished at $(date). Starting LHP-CZSL training on GPU 1..."

cd /home/dkkim/.gemini/antigravity/scratch/LHP-CZSL
CUDA_VISIBLE_DEVICES=1 python train.py --yml_path config/lhp_czsl_mit.yml
