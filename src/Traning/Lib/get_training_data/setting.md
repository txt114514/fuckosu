# setting.json 使用说明

这份文档说明 `src/Traning/Lib/get_training_data/setting.json` 的用途和写法。

`setting.json` 负责控制“这次要不要跑某个阶段、是否覆盖旧结果、是否继续执行、是否启用音频对齐匹配、人工微调偏移是多少”。

它和 `config.json` 的分工是：

- `config.json`：放路径、文件名、处理参数、阶段依赖这类“结构配置”
- `setting.json`：放本次运行要不要执行某一步这类“运行开关”

## 1. 默认加载方式

直接运行下面命令时：

```bash
python3 src/Traning/Lib/get_training_data/training_main.py
```

`training_main.py` 会默认读取同目录下的 `setting.json`。

如果你想改用别的设置文件，可以传：

```bash
python3 src/Traning/Lib/get_training_data/training_main.py --setting /path/to/setting.json
```

## 2. 当前完整格式

当前 `setting.json` 的标准格式如下：

```json
{
    "runtime": {
        "overwrite": false,
        "continue_on_error": false
    },
    "check_data": {
        "enabled": true,
        "run_get_files": true,
        "run_verify_export": true,
        "run_difficulty_export": true
    },
    "video_clip": {
        "enabled": true,
        "run_video_init_check": true,
        "run_video_match": true,
        "run_av_correspondence": true,
        "run_clip_stage": true,
        "use_audio_match_experiment": true,
        "global_offset_ms": 0.0
    }
}
```

## 3. 字段说明

### `runtime`

- `overwrite`
  含义：是否覆盖已经处理完成的结果。
  `true`：强制重跑并覆盖旧输出。
  `false`：已完成的阶段可能会被跳过。

- `continue_on_error`
  含义：某个顶层阶段失败后，是否继续跑下一个顶层阶段。
  `true`：失败后继续。
  `false`：遇到错误就停止。

### `check_data`

- `enabled`
  含义：是否运行整个 `check_data` 阶段。
  `false` 时，这一组下面的其他开关都会失去实际作用。

- `run_get_files`
  含义：是否导入 `.osz/.osu`、重建 `order.txt`、导入 `audio.mp3`。

- `run_verify_export`
  含义：是否导出 `verify.txt`。

- `run_difficulty_export`
  含义：是否导出 `difficulty.txt`。

### `video_clip`

- `enabled`
  含义：是否运行整个 `video_clip` 阶段。
  `false` 时，这一组下面的其他开关都会失去实际作用。

- `run_video_init_check`
  含义：是否先做视频流程初始化与状态同步。

- `run_video_match`
  含义：是否执行视频匹配。

- `run_av_correspondence`
  含义：是否执行 AV 对齐并输出 `video_processed.mp4`。

- `run_clip_stage`
  含义：是否执行最后的固定区域裁剪。

- `use_audio_match_experiment`
  含义：是否使用“音频对齐 + verify”方式做视频匹配。
  `true`：启用新版匹配方式。
  `false`：退回顺序/文件名规则匹配方式。

  说明：
  现在默认值就是 `true`，也就是默认使用“音频对齐 + verify”。

- `global_offset_ms`
  含义：人工全局微调偏移，单位毫秒。
  这个值会在“音频对齐 + verify 修正”之后再叠加。

  例子：
  - `0.0`：不做额外人工微调
  - `5.0`：整体再往后推 5ms
  - `-8.0`：整体再往前拉 8ms

## 4. 推荐理解方式

最终 AV 偏移的思路是：

1. 先做音频对齐
2. 再用 `verify.txt` 修正鼓点位置
3. 最后叠加 `global_offset_ms` 做人工微调

所以一般建议：

- 先保持 `use_audio_match_experiment = true`
- 先把 `global_offset_ms` 设成 `0.0`
- 只有在人工复核后确认仍有整体稳定偏差时，再改 `global_offset_ms`

## 5. 常见配置示例

### 例 1：跑完整流程，使用新版音频对齐

```json
{
    "runtime": {
        "overwrite": false,
        "continue_on_error": false
    },
    "check_data": {
        "enabled": true,
        "run_get_files": true,
        "run_verify_export": true,
        "run_difficulty_export": true
    },
    "video_clip": {
        "enabled": true,
        "run_video_init_check": true,
        "run_video_match": true,
        "run_av_correspondence": true,
        "run_clip_stage": true,
        "use_audio_match_experiment": true,
        "global_offset_ms": 0.0
    }
}
```

### 例 2：只重跑 AV 对齐和裁剪

```json
{
    "runtime": {
        "overwrite": true,
        "continue_on_error": false
    },
    "check_data": {
        "enabled": false,
        "run_get_files": false,
        "run_verify_export": false,
        "run_difficulty_export": false
    },
    "video_clip": {
        "enabled": true,
        "run_video_init_check": false,
        "run_video_match": false,
        "run_av_correspondence": true,
        "run_clip_stage": true,
        "use_audio_match_experiment": true,
        "global_offset_ms": 0.0
    }
}
```

### 例 3：只做人工微调后重跑视频阶段

```json
{
    "runtime": {
        "overwrite": true,
        "continue_on_error": false
    },
    "check_data": {
        "enabled": false,
        "run_get_files": false,
        "run_verify_export": false,
        "run_difficulty_export": false
    },
    "video_clip": {
        "enabled": true,
        "run_video_init_check": false,
        "run_video_match": false,
        "run_av_correspondence": true,
        "run_clip_stage": true,
        "use_audio_match_experiment": true,
        "global_offset_ms": -1.0
    }
}
```

## 6. 与命令行参数的关系

`training_main.py` 仍然支持命令行参数，这些参数会覆盖 `setting.json` 的部分行为。

常用的覆盖方式有：

- `--setting <path>`
  使用指定的设置文件。

- `--overwrite`
  临时覆盖 `runtime.overwrite=true`。

- `--continue-on-error`
  临时覆盖 `runtime.continue_on_error=true`。

- `--use-audio-match-experiment`
  临时强制启用音频对齐匹配。

- `--disable-audio-match-experiment`
  临时关闭音频对齐匹配。

- `--global-offset-ms <value>`
  临时覆盖 `video_clip.global_offset_ms`。

- `--skip-check-data`
- `--skip-get-files`
- `--skip-verify-export`
- `--skip-difficulty-export`
- `--skip-video-clip`
- `--skip-video-init`
- `--skip-video-match`
- `--skip-av-correspondence`
- `--skip-clip-stage`

这些 `skip-*` 参数的作用都是“本次命令临时关闭对应阶段”。

## 7. 注意事项

- `setting.json` 必须是标准 JSON，不能写注释。
- 布尔值只能写 `true` 或 `false`。
- `global_offset_ms` 必须是有限数字。
- 如果 `setting.json` 格式错误，程序会打印报错，并回退到代码默认运行设置。
- 如果你把某个顶层阶段的 `enabled` 设成 `false`，它下面的子开关即使是 `true` 也不会执行。

## 8. 建议

如果你的目标是稳定使用当前新版流程，建议直接保持：

- `video_clip.use_audio_match_experiment = true`
- `video_clip.global_offset_ms = 0.0`

然后只在下面两种情况下改动：

- 需要跳过某个阶段时，改对应的 `enabled` 或 `run_*`
- 人工复核后发现还有稳定整体偏移时，再调整 `global_offset_ms`
