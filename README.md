# coreldraw-vectorize-mcp

把位图（示意图、地质图、图表、流程图、表格、扫描件……）**一次调用**转成**可编辑的 CorelDRAW 矢量文件**的 MCP 服务器。
线条、色块、网格是矢量曲线，文字是可编辑的文本对象；模糊、缩小、JPEG 压缩过的图也能处理。
结果中拿不准的地方会**报告给调用它的 AI**，由 AI 核对修正后一键重建。

*An MCP server that turns raster drawings (diagrams, maps, charts, flowcharts, tables, scans) into editable
CorelDRAW vector files in one call: traced linework and colour regions, re-created editable text, and a
self-check that reports anything uncertain back to the calling AI so it can fix and rebuild.*

![10 张未调参的新图：左输入，右一次调用的结果](docs/overview.png)

<sub>10 张 MCP 开发时从未针对其调参的图：每组左为输入，右为一次调用、无人工修正的结果。</sub>

## 功能

- **一次调用**：`cdr_vectorize(image_path)` 完成超分辨率、OCR、分层描摹、文字重建，输出 `result.cdr` + `result.png`
- **自动判断模式**：彩色（色块/填充/网格分层）或黑白线稿
- **文字变成可编辑文本**：多尺度 OCR 投票；支持斜排、竖排文字；粘连的多个标签自动拆分；中文用新宋体，数字/英文按原图比对选 Times New Roman 或 Arial；按原图笔画粗细模拟加粗
- **适应低质量输入**：Real-ESRGAN 超分；大图自动缩放；扫描件纸张底色校正；JPEG 褪色色块按色相找回
- **细节保真**：网格线识别为矢量折线；PowerTRACE 会丢掉的点状填充、短剖面线直接画成矢量；色块深色描边还原
- **自检并报告问题**（`issues`）：图形覆盖率、多余内容、颜色、小元素丢失、描摹失败、文字回读；附差异图（红 = 缺失，蓝 = 多余）
- **核对图**（`review_sheet`）：把不确定的文字（L#）和可能漏识别的文字（M#）拼成一张图，AI 看一张图就能核对

![JPEG 压缩地图：原图 / 输入 / 早期版本 / 现在](docs/map_jpeg.png)

![缩小、模糊的测井图：点状填充的还原](docs/stipple.png)

## 环境要求

| 项 | 说明 |
|---|---|
| 系统 | Windows（通过 COM 驱动 CorelDRAW） |
| CorelDRAW | 已在 **CorelDRAW Graphics Suite 2022（v24）** 上验证；使用前需先手动打开 CorelDRAW 窗口 |
| Python | 3.11+ |
| Real-ESRGAN | [realesrgan-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN/releases) Windows 版（需要支持 Vulkan 的显卡） |

## 安装

```bash
git clone https://github.com/cjjqxx/coreldraw-vectorize-mcp.git
```

```bash
cd coreldraw-vectorize-mcp
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install -r requirements.txt
```

然后下载 [Real-ESRGAN ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN/releases)（`realesrgan-ncnn-vulkan-*-windows.zip`），解压成：

```
tools/realesrgan/realesrgan-ncnn-vulkan.exe
tools/realesrgan/models/realesrgan-x4plus-anime.*
```

（也可以放在别处，用环境变量 `CDR_TOOLS` 指向包含 `realesrgan/` 的目录。）

## 接入 MCP 客户端

在 Claude Code / Claude Desktop 等客户端的 MCP 配置中加入（路径改成你的）：

```json
{
  "mcpServers": {
    "cdr": {
      "type": "stdio",
      "command": "C:\\path\\to\\coreldraw-vectorize-mcp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\coreldraw-vectorize-mcp\\cdr_server.py"]
    }
  }
}
```

可选环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `CDR_WORK` | 未指定 `work_dir` 时的输出目录 | `~/cdr-vectorize-work` |
| `CDR_TOOLS` | 含 `realesrgan/` 的目录 | 仓库下的 `tools/` |
| `CDR_EXE` | CorelDRAW 可执行文件（仅 `cdr_launch` 使用） | 默认安装路径 |

## 使用

### 推荐流程

1. 打开 CorelDRAW。
2. 调用 `cdr_vectorize(image_path)`。返回 `result.cdr`、`result.png`，以及：
   - `needs_review`：是否有需要核对的地方
   - `labels_to_check`：不确定的文字（原因 + 其他可能读法）
   - `unlabeled_text`：疑似没识别出来的文字区域
   - `review_sheet`：一张核对图
   - `issues`：图形自检发现的问题（带区域坐标和修复建议），`self_check.diff_png` 为差异图
3. 若 `needs_review` 为 true：AI 查看 `review_sheet` / `diff_png`，修正 `labels`，调用
   `cdr_vectorize_build(work_dir, labels=修正后的列表)` 重建——超分和 OCR 结果复用，只重新分层、描摹、放文字。

### 工具列表

| 工具 | 用途 |
|---|---|
| `cdr_vectorize` | 一次调用完成图片 → CDR（推荐） |
| `cdr_vectorize_prepare` / `cdr_vectorize_build` | 分两步：先超分 + OCR，再按（修正后的）标签生成 |
| `cdr_trace_image`、`cdr_reproduce`、`cdr_compare` | 纯 OpenCV 轮廓描摹（不依赖 PowerTRACE）及对比 |
| `cdr_status`、`cdr_launch`、`cdr_new_document`、`cdr_draw_*`、`cdr_add_text`、`cdr_import_image`、`cdr_export_png`、`cdr_save_document` | CorelDRAW 基础自动化 |

## 已知限制

- 中文字体统一用新宋体（按原图自动区分宋体/黑体在常见分辨率下不可靠）。
- 标题中的连续空格会丢失（OCR 结果不含空格）。
- 与坐标刻度线相连的小数字可能偏大；这类情况会作为 `text_mismatch` 报告。
- 字高只有六七个像素的极小文字无法可靠识别，会列入核对清单。
- 仅支持 Windows + CorelDRAW；其他 CorelDRAW 版本未测试。

## 测试

`regress/` 下有两套测试，都直接调用 MCP 的一次调用流程，不做人工修正：

```bash
.venv\Scripts\python regress\vec_regress.py run mytest
```

回归集：2 张参考图 × 原图 / 缩小 / 模糊 / JPEG，按标准答案打分（色块、墨线、网格、文字、丢失元素等）；`compare` 子命令对比两次运行，任何指标变差都会标出。

```bash
.venv\Scripts\python regress\holdout_run.py mytest
```

新图集：10 张开发时从未调参的图，用 MCP 自检结果评估。`regress/make_synthetic.py` 可重新生成其中 4 张合成图（需要 matplotlib）。

## 测试图片来源

- `regress/holdout/g_*.png`：由 `make_synthetic.py` 生成。
- `regress/holdout/h_*.png`、`s_*.png`：作者自己的研究配图。
- `regress/refs/tibet_geo.png`、`well_logging.png`：来自公开文献/资料，仅用于测试。如您是版权方并希望移除，请提 issue，会立即删除。

## 许可证

[Apache License 2.0](LICENSE)。Real-ESRGAN 与 CorelDRAW 为第三方软件，不包含在本仓库中，遵循其各自的许可条款。
