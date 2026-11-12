# 接触线反馈探测

保留现有双帧 SE(3) 模型。默认 `--contact-line-strategy feedback` 使用同步帧、触觉形变反馈 rocking、表面接触线有效性检查，然后在接触线连续稳定后沿线摆动。探测轨迹参考原项目：先在 5 秒内将 roll 目标渐增到 1°，再按 20 秒周期做 roll 1°、pitch 0.6° 的椭圆 rocking，默认持续 1.25 周期；无有效线时平滑释放。实际 TCP 仍受速度、加速度、单步和相对初始姿态最大 4°约束，法向总位移默认限 1 mm。形变反馈不等同于恒力控制。

## 标准位置与接触平面

按当前项目的标准设置，未传 `--contact-line-plane-point-mm` 时，程序读取 `--standard-tcp-to-nut-center` 并将米转换为毫米。当前标准值 `[0, 0, 0.100] m` 对应默认平面点 `[0, 0, 100] mm`，法向默认 `[0, 0, 1]`。如果修改标准偏移，默认平面点随之更新。

这是按标准螺母中心位置近似接触平面。代码没有单独的螺母厚度或底面偏移标定，因此该默认值不表示实际底面已经完成标定；这个偏差会影响接触线的位置精度。启动日志与 metadata 会记录 `standard_tcp_to_nut_center_approximation`。显式传入 `--contact-line-plane-point-mm X Y Z` 时，以该值为准，记录 `explicit_plane_point_mm`。

点和法向均使用探测参考 TCP 的局部坐标系，位置单位为毫米，不能填写机器人基座坐标。程序利用实际 TCP 姿态转换到基座系。`--bolt-contact-tcp-pose` 是机器人 TCP 的基座位姿，继续用于原有准备/下降流程，不是该平面的局部坐标。

## 运行

如果螺母已夹持并已与螺栓顶部/边缘建立轻接触：

```bash
python tactile_policy/infer_tactile_servo_nutbolt_contact_line.py \
  --mode contact-line-probe \
  --checkpoint tactile_policy/outputs/relative_se3_200_217/best_model.pt \
  --no-contact-line-auto-prepare \
  --contact-line-contact-established
```

不传 `--execute` 时只做现有观察检查。实际执行在相同命令后添加 `--execute`，程序保留原有 RUN 确认。`--contact-line-contact-established` 表示操作者确认当前存在外部接触；日志会注明这是人工依据。也可以传 `--contact-line-free-grasp-forces-n LEFT RIGHT`，用相同夹持条件下、接触螺栓前的双侧力范数作为基准，持续检查力变化。该力变化判断是启发式接触依据。

需要自动夹持、搬运和下降时，使用原有绝对位姿模型作为准备模型：

```bash
python tactile_policy/infer_tactile_servo_nutbolt_contact_line.py \
  --mode contact-line-probe \
  --checkpoint tactile_policy/outputs/relative_se3_200_217/best_model.pt \
  --preparation-checkpoint /path/to/absolute_pose/best_model.pt
```

准备流程保留原来的标准抓取、螺栓接触 TCP 位姿和高度窗口，下降前测得的自由夹持力基准交给新探测流程。上面的准备模型路径需要换成实际文件。

## 有效性与记录

左右帧捕获时间差默认不超过 20 ms；同时检查帧龄和 TCP 插值时间范围。重复帧、失去同步或时钟回跳不会被当作新测量继续积分。TCP 缓存轮询目标为 100 Hz，模型/控制默认 20 Hz；实际频率取决于通信与推理延迟。

表面估计器允许平面内滑动，检查法向激励、公共线可观测性、原始平面残差和求解时间。几何有效不等于外部接触已被独立证实。默认至少完成一周期 rocking，且接触线连续稳定后才进入 follow；rocking 总时长由 `--contact-line-rock-cycles` 控制，支持小数周期。follow 中线过时或失效会停止。正常结束时将形变目标平滑归零，不直接回初始 TCP。

默认反馈 rocking 幅值为 roll 1°、pitch 0.6°，周期 20 秒、1.25 周期；可用 `--contact-line-deformation-roll-deg`、`--contact-line-deformation-pitch-deg`、`--contact-line-rock-period-s` 和 `--contact-line-rock-cycles` 覆盖。反馈点仍使用本项目当前 TCP 到螺母中心的配置；原项目的 `l_approx=54.5 mm` 定义在 ABB 的 cart/TCP 偏置坐标中，不可直接当作 UR5 TCP 偏移复用。默认写入 `tactile_policy/logs/contact_line_probe.csv` 与同名 `.metadata.json`，记录同步时间差、形变目标与误差、当前/指令 TCP、接触依据、线有效性和退化原因。`--contact-line-no-follow` 可只采集反馈 rocking；`--contact-line-strategy waypoints --contact-line-estimator svd` 可使用旧航点探测用于对比。

力变化接触判断采用迟滞：`--contact-line-contact-change-n` 默认 0.5 N，用于建立接触依据；`--contact-line-contact-release-n` 默认 0.25 N，用于建立后的维持判断，必须小于前者。弱依据期间立即暂停命令；至少 3 个连续新帧且捕获时间跨度达到 `--contact-line-contact-loss-s`（默认 0.3 秒）才报丢失。超过 0.2 秒的采样间隔会重新开始连续低值计数，缺帧仍由原有同步/反馈超时检查处理。该迟滞减少阈值附近的反复切换，不代表能独立证明螺母没有脱离螺栓。

CSV 的 `event=control` 表示一次控制记录；`waiting_for_sync`、`contact_evidence` 和 `contact_evidence_lost` 记录暂停或停止。这些事件行可能没有模型输出，分析时应按 `event` 筛选；`motion_command_sent` 表示该记录是否将发送运动命令。`contact_strength_n` 是双侧力变化绝对值的最大值，`contact_low_s` 与 `contact_low_samples` 记录连续弱依据时间和帧数。异常信息和 metadata 同时保存停止时的双侧力、自由夹持基准和力变化。

表面优化使用解析稀疏 Jacobian，并限制内部线性迭代；原有 0.2 秒求解预算、收敛检查、几何可观测性和残差门限继续生效。求解加速不会把几何退化的窗口改判为有效。

新增控制、同步与表面估计已通过离线单元测试；实际机器人行为和接触线精度仍需实机验证。
