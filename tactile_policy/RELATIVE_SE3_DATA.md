# Paired tactile local-SE(3) data

Each automatic `L` cycle in
`real_ur5rg2/collect_lerobot_xenstactile_contactline.py` now performs:

1. sample the existing calibrated-nut-relative open-gripper pose;
2. move there, close RG2, and wait for contact to settle;
3. generate one clipped and moving-average-smoothed OU trajectory;
4. save one synchronized LeRobot frame for every valid OU step;
5. save the whole trajectory as one episode, open RG2, and repeat.

Use a **new dataset root** because the schema contains three new fields:

- `observation.tcp_pose`: actual synchronized UR TCP `[x,y,z,rx,ry,rz]`;
- `observation.ou_offset`: commanded local OU offset;
- `observation.episode_initial_tcp_pose`: TCP at the start of the OU trajectory.

The existing left/right distributed force, force6d, depth, rectify images,
sensor timestamps, camera images, robot state and force/torque fields remain.

Example collection (after starting `launch_nodes.py` as before):

```bash
python real_ur5rg2/collect_lerobot_xenstactile_contactline.py \
  --root real_ur5rg2/data/nut_relative_ou \
  --num-episodes 100 \
  --fps 20 \
  --ou-frames 800
```

Press `L` once to start the automatic repeated collection sequence.

OU commands now use the calibrated fingertip centre as their rotation pivot:
`T_world_tcp_target = T_world_tcp_initial @ T_tcp_fingertip @ Delta @ inverse(T_tcp_fingertip)`.
Default local translation limits are ±(0.25, 0.5, 0.5) mm, and rotation-vector
component limits are ±(1, 1, 1.2) degrees. Rotation innovation amplitudes were
scaled with the expanded limits. These are clipping limits, not guaranteed
trajectory excursions. Pure rotation keeps the fingertip centre stationary;
the TCP moves along an arc. Saved `observation.ou_offset` now refers to this
initial fingertip frame. Use a new dataset root to distinguish this convention.

Training labels remain actual TCP-relative transforms. Because the TCP moves
around the fingertip pivot, their translation can exceed the OU translation
limits. The training script's old `--target-limit` defaults are therefore not
appropriate for this collection: determine limits from actual paired labels
(including both endpoints), or use an unbounded regression output before
training on these trajectories.

Train the paired model:

```bash
python tactile_policy/train_relative_se3.py --dataset-root /media/zhou/Elements SE/YBZHOU/nut_relative_ou_no_fts --output-dir tactile_policy/outputs/nut_relative_se3 --pairs-per-episode 10
```

Training and validation are split by complete episodes. For every selected
pair, the label is computed from actual robot feedback:

```text
T_1_to_2 = inverse(T_world_tcp_1) @ T_world_tcp_2
label = [local_x, local_y, local_z, local_rotation_vector]
```

The network input order is `[left_1, left_2, right_1, right_2]`, using one
shared tactile encoder, as in the reference fixed-base FC method. The native
XENSE `35x20` force grid is not resized.

python real_ur5rg2/merge_lerobot_datasets.py \
  --source "/media/zhou/Elements SE/YBZHOU/nut_relative_ou" \
  --source "/media/zhou/Elements SE/YBZHOU/nut_relative_ou_no_fts_test2" \
  --output "/media/zhou/Elements SE/YBZHOU/nut_relative_ou_217"