#!/bin/bash
gpu=$1
cfg=$2
dataset=$3

# Create logs directory if not exists
mkdir -p logs

# 修改logs的路径为logs/「中间加一个文件夹，包含时间戳」/logfile
# 终端输出的结果同样也要保存在logfile里面

timestamp=$(date +%Y%m%d_%H%M%S)
logs_dir="logs/training_${timestamp}"
mkdir -p ${logs_dir}
# scenes=(aloe art century flowers garbage picnic roses)
scenes=(curated_d_t1) # curated_c1_a_t1 curated_c1_a_t2 curated_c1_b_t1 curated_c1_b_t2 curated_c2_a_t1 curated_c2_a_t2 curated_c2_b_t1 curated_c2_b_t2
for scene in "${scenes[@]}"; do
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$start_time] Starting stage1 for scene: $scene" >> ${logs_dir}/${scene}_stage1.log
    python ../feishu_notify.py "🚀 Starting stage1 training for scene: $scene"
    
    if bash scripts/run_stage1.sh ${scene} ${gpu} ${cfg} ${dataset} 1 >> ${logs_dir}/${scene}_stage1.log 2>&1; then
        end_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$end_time] Completed stage1 for scene: $scene" >> ${logs_dir}/${scene}_stage1.log
        python ../feishu_notify.py "✅ Completed stage1 training for scene: $scene"
    else
        fail_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$fail_time] Failed stage1 for scene: $scene" >> ${logs_dir}/${scene}_stage1.log
        python ../feishu_notify.py "❌ Failed stage1 training for scene: $scene"
    fi
done

# scenes=(pipe table)
# for scene in "${scenes[@]}"; do
#     start_time=$(date '+%Y-%m-%d %H:%M:%S')
#     echo "[$start_time] Starting stage1 for scene: $scene" >> ${logs_dir}/${scene}_stage1.log
#     python ../feishu_notify.py "🚀 Starting stage1 training for scene: $scene"
    
#     if bash scripts/run_stage1.sh ${scene} ${gpu} ${cfg} ${dataset} 0 >> ${logs_dir}/${scene}_stage1.log 2>&1; then
#         end_time=$(date '+%Y-%m-%d %H:%M:%S')
#         echo "[$end_time] Completed stage1 for scene: $scene" >> ${logs_dir}/${scene}_stage1.log
#         python ../feishu_notify.py "✅ Completed stage1 training for scene: $scene"
#     else
#         fail_time=$(date '+%Y-%m-%d %H:%M:%S')
#         echo "[$fail_time] Failed stage1 for scene: $scene" >> ${logs_dir}/${scene}_stage1.log
#         python ../feishu_notify.py "❌ Failed stage1 training for scene: $scene"
#     fi
# done

echo "Sweep nerfbusters done!"