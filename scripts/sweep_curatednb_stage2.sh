#!/bin/bash
gpu=$1
cd=$2
cs=$3
cg=$4
cu=$5
cv=$6
ckpt=$7
ckpt_iter=$8
dataset=$9
logname=${10}
expname=${11}

# Create logs directory if not exists
mkdir -p logs

timestamp=$(date +%Y%m%d_%H%M%S)
logs_dir="logs/training_${timestamp}"
mkdir -p ${logs_dir}

# scenes=(aloe art century flowers garbage picnic roses)
# scenes=(aloe century flowers garbage picnic roses)
scenes=(curated_c_t1_I)
for scene in "${scenes[@]}"; do
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$start_time] Starting stage2 for scene: $scene" >> ${logs_dir}/${scene}_stage2.log
    python ../feishu_notify.py "🚀 Starting stage2 training for scene: $scene"
    
    if bash scripts/run_stage2.sh ${scene} ${gpu} ${cd} ${cs} ${cg} ${cu} ${cv} ${ckpt} ${ckpt_iter} ${dataset} ${logname} 1 ${expname} >> ${logs_dir}/${scene}_stage2.log 2>&1; then
        end_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$end_time] Completed stage2 for scene: $scene" >> ${logs_dir}/${scene}_stage2.log
        python ../feishu_notify.py "✅ Completed stage2 training for scene: $scene"
    else
        fail_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$fail_time] Failed stage2 for scene: $scene" >> ${logs_dir}/${scene}_stage2.log
        python ../feishu_notify.py "❌ Failed stage2 training for scene: $scene"
    fi
done

# scenes=(pipe table)
# for scene in "${scenes[@]}"; do
#     start_time=$(date '+%Y-%m-%d %H:%M:%S')
#     echo "[$start_time] Starting stage2 for scene: $scene" >> ${logs_dir}/${scene}_stage2.log
#     python ../feishu_notify.py "🚀 Starting stage2 training for scene: $scene"
    
#     if bash scripts/run_stage2.sh ${scene} ${gpu} ${cd} ${cs} ${cg} ${cu} ${cv} ${ckpt} ${ckpt_iter} ${dataset} ${logname} ${expname} >> ${logs_dir}/${scene}_stage2.log 2>&1; then
#         end_time=$(date '+%Y-%m-%d %H:%M:%S')
#         echo "[$end_time] Completed stage2 training for scene: $scene" >> ${logs_dir}/${scene}_stage2.log
#         python ../feishu_notify.py "✅ Completed stage2 training for scene: $scene"
#     else
#         fail_time=$(date '+%Y-%m-%d %H:%M:%S')
#         echo "[$fail_time] Failed stage2 for scene: $scene" >> ${logs_dir}/${scene}_stage2.log
#         python ../feishu_notify.py "❌ Failed stage2 training for scene: $scene"
#     fi
# done

echo "Sweep nerfbusters done!"
