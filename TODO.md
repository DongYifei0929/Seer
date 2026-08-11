现在我已经加了一个 Seer 专用桥接脚本：

  scripts/export_seer_actions_for_robolab.py

  它做的事是：读取 RoboLab episode 里的 rgb.mp4 + wrist.mp4 + data.parquet，调
  用 InternRobotics/Seer 的 SeerController，导出 RoboLab rel_ik 可回放的 7D
  action .npz。没有调用 GR00T。

  运行方式：
```bash
  cd /mnt/afs/dongyifei/DreamFlyWheel/GR00T-Dreams

  python scripts/export_seer_actions_for_robolab.py \
    --seer-root /mnt/afs/dongyifei/DreamFlyWheel/Seer \
    --dataset-dir /mnt/afs/dongyifei/DreamFlyWheel/RoboLab/output/robolab_with_depth/BananaInBowlTask_env_000 \
    --resume-from-checkpoint /mnt/afs/dongyifei/DreamFlyWheel/Seer/checkpoints/real_world_droid/seer.pth \
    --vit-checkpoint-path /mnt/afs/dongyifei/DreamFlyWheel/Seer/checkpoints/vit_mae/mae_pretrain_vit_base.pth \
    --output outputs/seer_idm/BananaInBowlTask_env_000/robolab_actions_rel_ik.npz
```
  然后在 RoboLab 里部署回放：
```bash
  cd /mnt/afs/dongyifei/DreamFlyWheel/RoboLab

  uv run python scripts/idm_deploy/deploy_idm_actions.py \
    --actions /mnt/afs/dongyifei/DreamFlyWheel/GR00T-Dreams/outputs/seer_idm/BananaInBowlTask_env_000/robolab_actions_rel_ik.npz \
    --task BananaInBowlTask \
    --controller rel_ik \
    --source-dataset output/robolab_with_depth/BananaInBowlTask_env_000 \
    --reference-video output/robolab_with_depth/BananaInBowlTask_env_000/rgb.mp4 \
    --enable-subtask \
    --headless
```
  当前还没法实际跑 Seer 推理，因为本机现有环境缺 Seer 依赖和权重：clip、wandb
  等没装，并且没找到 mae_pretrain_vit_base.pth 或 Seer Droid checkpoint。Seer
  官方权重入口在 README 的 Checkpoints
  部分：https://github.com/InternRobotics/Seer
  (https://github.com/InternRobotics/Seer)。下载安装到 Seer
  环境后，上面命令就能走 Seer IDM 路线。

* Seer 输出
所以如果你说“纯 Seer 的预测”，对机械臂部分可以理解成 eef delta；完整输出还要加gripper。更细一点说，它通常先是在归一化动作空间里预测，物理单位要在控制/后处理里还原。real_controller/controller.py
==目前数据在相机坐标系下，脚本应该也是相机坐标系下的==

TODO
Benchmark 构建 （依托RoboLab？统计任务分布）
droid上离线测试一下Seer，统计误差