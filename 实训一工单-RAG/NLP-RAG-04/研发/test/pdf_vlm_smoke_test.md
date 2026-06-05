# PDF VLM 单页冒烟测试

这个小测试不是重建整个知识库，而是单独拿一页 PDF 去调用一次 `PDF VLM`，用来定位：

- 接口是否真正返回内容
- 返回内容是不是标准 JSON
- 为什么 `vlm_api_success_pages` 有值，但 `vlm_enhanced_chunks = 0`
- 本地乱码文本是否把 VLM 带偏

## 适用场景

当你发现：

- `enhance_ingest.bat` 很慢
- `page_*.json` 只有 `[]`
- `page_*.raw.json` 想单独检查

就先跑这个测试，不用整库重建。

## 启动前提

1. 后端环境变量要和 `start_backend_local.bat` 一致，尤其是：
   - `PDF_VLM_API_KEY`
   - `PDF_VLM_API_URL`
   - `PDF_VLM_MODEL_NAME`
2. 推荐在 `nlp-rag` 环境执行
3. `parse2` 相关依赖要可用，因为脚本会先走一次本地解析

## 执行命令

在项目根目录执行：

```powershell
cd D:\Desktop\NLP-RAG-02
conda activate nlp-rag
set PDF_PARSER_BACKEND=parse2
set PDF_PARSER_PYTHON=D:\anaconda2024\envs\pdf-parser\python.exe
set PDF_VLM_API_KEY=你的key
python test\pdf_vlm_smoke_test.py --page 23
```

这个脚本已经内置了项目根目录注入，不需要你额外设置 `PYTHONPATH`。

如果你想指定 PDF：

```powershell
python test\pdf_vlm_smoke_test.py --page 23 --pdf D:\Desktop\NLP-RAG-02\data\prospectus.pdf
```

如果你怀疑图片渲染太小，也可以提高渲染倍率：

```powershell
python test\pdf_vlm_smoke_test.py --page 23 --render-scale 1.8
```

如果你想排除“本地乱码文本干扰”，可以只让模型看图片：

```powershell
python test\pdf_vlm_smoke_test.py --page 4 --mode image_only
```

如果你想看模型在“强制尽量输出”时会不会给出结果：

```powershell
python test\pdf_vlm_smoke_test.py --page 4 --mode image_only --force-items
```

## 输出文件

脚本会在下面目录生成结果：

```text
artifacts/pdf_vlm_smoke_test/prospectus/
```

重点看三个文件：

1. `page_23.raw.json`
   - 模型原始返回
   - 最重要，用来判断接口到底回了什么

2. `page_23.json`
   - 解析后的结构化结果
   - 如果这里只有 `[]`，而 `raw.json` 里明明有内容，说明是解析器问题

3. `page_23.summary.json`
   - 本地文本预览、表格预览、状态、条目数量
   - 方便快速判断这一页值不值得送 VLM

如果你用的是 `image_only` 模式，对应文件会变成：

- `page_4.image_only.raw.json`
- `page_4.image_only.json`
- `page_4.image_only.summary.json`

如果再加 `--force-items`，文件会变成：

- `page_4.image_only.force.raw.json`
- `page_4.image_only.force.json`
- `page_4.image_only.force.summary.json`

## 怎么判断问题在哪

### 情况 1：`raw.json` 里就是 `[]`

说明模型本身判断“这页没有高价值补充内容”。

常见原因：

- prompt 太保守
- 本地文本已经把关键信息说全了
- 这页对多模态模型来说信息密度不够

### 情况 2：`raw.json` 里有自然语言，但 `page_23.json` 是 `[]`

说明接口通了，但模型没有严格按 JSON 输出。

这时该优化：

- prompt
- JSON 解析容错

### 情况 3：`raw.json` 里有 JSON，但 `page_23.json` 还是 `[]`

说明结构字段不符合当前解析器要求，比如：

- 缺少 `title`
- 缺少 `value`
- 字段名不一致

### 情况 4：直接报超时

说明是在线接口性能问题，不是解析问题。

### 情况 5：`full` 模式是乱码，`image_only` 模式变正常

说明问题不在 VLM，而在本地解析文本质量。
这时应该优先修本地解析链路，或者在增强阶段让 VLM 更依赖图片本身。

## 推荐先测哪些页

优先测这些页：

- 23
- 30
- 32
- 33
- 37
- 38
- 42
- 52
- 53

因为这些页在你之前的增强结果里属于 `vlm_api_success_pages`。

## 结论目标

这个测试的目标不是“让 VLM 立刻变强”，而是先回答这个关键问题：

**PDF VLM 没通，到底是因为模型没返回、返回格式不对，还是我们解析器没接住。**
