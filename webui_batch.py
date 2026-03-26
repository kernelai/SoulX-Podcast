"""
Batch synthesis Tab for Gradio WebUI.

Provides file upload, task queue management, and batch download
as a separate Tab alongside the existing single-synthesis interface.
"""
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import gradio as gr
import numpy as np
import random
import torch

from batch.file_parser import parse_txt_files
from batch.task_queue import BatchTaskQueue, SpeakerConfig, STATUS_COMPLETED
from batch.downloader import pack_to_zip

logger = logging.getLogger(__name__)


def concat_generated_wavs(results_dict: dict) -> np.ndarray:
    """
    Concatenate generated wav segments into a single audio array.

    Shared by both single-synthesis and batch-synthesis paths to
    avoid logic duplication.

    Args:
        results_dict: Output from model.forward_longform(),
                      must contain 'generated_wavs' key.

    Returns:
        1-D numpy float32 array of the concatenated audio.
    """
    wavs = results_dict["generated_wavs"]
    if not wavs:
        return np.array([], dtype=np.float32)
    target_audio = wavs[0]
    for wav in wavs[1:]:
        target_audio = torch.concat([target_audio, wav], axis=1)
    return target_audio.cpu().squeeze(0).numpy()

# Batch output directory: output/batch_{timestamp}/
_BATCH_OUTPUT_BASE = Path("output/batch")


def _create_batch_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _BATCH_OUTPUT_BASE / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


# Global batch queue instance (initialized lazily)
_batch_queue: Optional[BatchTaskQueue] = None
_batch_queue_lock = threading.Lock()


def _make_synthesize_fn(
    process_single_fn,
    get_model_fn: Callable,
) -> Callable[[str, SpeakerConfig], tuple[int, np.ndarray]]:
    """
    Create a synthesize callback compatible with BatchTaskQueue.

    Wraps process_single + model.forward_longform into
    (dialogue_text, SpeakerConfig) -> (sample_rate, audio_array).

    Args:
        process_single_fn: The process_single function from webui.py.
        get_model_fn: Callable that returns the loaded model (lazy reference
                      to avoid capturing None at import time).

    Note: model_lock is acquired by BatchTaskQueue._process_task,
    so this function does NOT acquire it again.
    """

    def synthesize(dialogue_text: str, config: SpeakerConfig) -> tuple[int, np.ndarray]:
        # Set seed
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Parse dialogue text into target_text_list
        target_text_list = re.findall(r"(\[S[0-9]\][^\[\]]*)", dialogue_text)
        target_text_list = [t.strip() for t in target_text_list]

        prompt_wav_list = [config.spk1_prompt_audio, config.spk2_prompt_audio]
        prompt_text_list = [config.spk1_prompt_text, config.spk2_prompt_text]
        use_dialect_prompt = (
            config.spk1_dialect_prompt_text.strip() != ""
            or config.spk2_dialect_prompt_text.strip() != ""
        )
        dialect_prompt_text_list = [
            config.spk1_dialect_prompt_text,
            config.spk2_dialect_prompt_text,
        ]

        data = process_single_fn(
            target_text_list,
            prompt_wav_list,
            prompt_text_list,
            use_dialect_prompt,
            dialect_prompt_text_list,
        )

        current_model = get_model_fn()
        results_dict = current_model.forward_longform(**data)

        audio_array = concat_generated_wavs(results_dict)
        return (24000, audio_array)

    return synthesize


def render_batch_tab(
    process_single_fn: Callable,
    get_model_fn: Callable,
    model_lock: threading.Lock,
) -> None:
    """
    Render the batch synthesis Tab inside an existing gr.Blocks context.

    Args:
        process_single_fn: The process_single function from webui.py.
        get_model_fn: Callable that returns the loaded model instance
                      (lazy reference to avoid capturing None at init time).
        model_lock: Global threading lock shared with single-synthesis Tab.
    """
    synthesize_fn = _make_synthesize_fn(process_single_fn, get_model_fn)

    with gr.Column():
        gr.Markdown("### 批量合成 - 上传 TXT 文件进行批量语音合成")

        # ---- Speaker config section ----
        with gr.Accordion("说话人配置 (全局)", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_spk1_audio = gr.Audio(
                        label="说话人 1 参考语音",
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    batch_spk1_text = gr.Textbox(
                        label="说话人 1 参考文本", lines=2
                    )
                    batch_spk1_dialect = gr.Textbox(
                        label="说话人 1 方言提示文本", lines=2
                    )
                with gr.Column(scale=1):
                    batch_spk2_audio = gr.Audio(
                        label="说话人 2 参考语音",
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    batch_spk2_text = gr.Textbox(
                        label="说话人 2 参考文本", lines=2
                    )
                    batch_spk2_dialect = gr.Textbox(
                        label="说话人 2 方言提示文本", lines=2
                    )

        # ---- File upload section ----
        with gr.Row():
            batch_files = gr.File(
                label="上传 TXT 文件（支持多文件）",
                file_count="multiple",
                file_types=[".txt"],
                interactive=True,
                scale=3,
            )
            with gr.Column(scale=1):
                batch_seed = gr.Number(label="Seed", value=1988, step=1)
                batch_start_btn = gr.Button(
                    "开始批量合成", variant="primary", size="lg"
                )
                batch_cancel_btn = gr.Button(
                    "取消全部排队任务", variant="stop", size="lg"
                )

        # ---- Task queue status ----
        gr.Markdown("### 任务队列")
        batch_summary = gr.Textbox(
            label="总进度", value="无任务", interactive=False
        )
        batch_table = gr.Dataframe(
            headers=["文件名", "状态", "进度", "耗时", "错误信息"],
            datatype=["str", "str", "str", "str", "str"],
            label="任务列表",
            interactive=False,
            row_count=(1, "dynamic"),
        )

        # ---- Download section ----
        gr.Markdown("### 下载与清理")
        with gr.Row():
            batch_zip_btn = gr.Button("打包下载 ZIP", scale=1)
            batch_delete_btn = gr.Button("清空已完成任务及文件", variant="stop", scale=1)
        batch_zip_file = gr.File(label="ZIP 下载", interactive=False)
        batch_single_select = gr.Dropdown(
            label="选择单个文件下载", choices=[], interactive=True
        )
        batch_single_file = gr.File(label="单文件下载", interactive=False)

        # Auto-refresh timer (every 3 seconds)
        batch_timer = gr.Timer(value=3, active=False)

        # ---- State to track whether queue is active ----
        batch_active = gr.State(value=False)

    # ---- Event handlers ----

    def on_start_batch(
        files,
        spk1_audio, spk1_text, spk1_dialect,
        spk2_audio, spk2_text, spk2_dialect,
        seed,
    ):
        """Handle batch start button click."""
        if not files:
            gr.Warning("请先上传 TXT 文件")
            return gr.update(), gr.update(), gr.update(), False

        global _batch_queue
        # Reset queue with new output dir for each batch run
        with _batch_queue_lock:
            if _batch_queue is not None and _batch_queue.is_busy():
                gr.Warning("上一个批次仍在处理中，请等待完成或取消后再试")
                return gr.update(), gr.update(), gr.update(), True

            # Shutdown old worker thread before creating new queue
            if _batch_queue is not None:
                _batch_queue.shutdown()

            output_dir = _create_batch_output_dir()
            _batch_queue = BatchTaskQueue(
                output_dir=output_dir,
                synthesize_fn=synthesize_fn,
                model_lock=model_lock,
            )
            _batch_queue.start_worker()

        q = _batch_queue

        # Parse files
        file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
        scripts = parse_txt_files(file_paths)

        # Build speaker config
        config = SpeakerConfig(
            spk1_prompt_audio=spk1_audio,
            spk1_prompt_text=spk1_text or "",
            spk1_dialect_prompt_text=spk1_dialect or "",
            spk2_prompt_audio=spk2_audio,
            spk2_prompt_text=spk2_text or "",
            spk2_dialect_prompt_text=spk2_dialect or "",
            seed=int(seed),
        )

        q.add_tasks(scripts, config)

        # Show initial table
        table = q.get_status_table()
        summary = q.get_summary()

        # Count invalid files
        invalid_count = sum(1 for s in scripts if not s.is_valid)
        if invalid_count > 0:
            gr.Warning(f"{invalid_count} 个文件格式错误，已跳过合成")

        return table, summary, gr.update(active=True), True

    def on_cancel():
        """Handle cancel button click."""
        q = _batch_queue
        if q is None:
            return gr.update(), gr.update()
        count = q.cancel_all_pending()
        if count > 0:
            gr.Info(f"已取消 {count} 个排队任务")
        return q.get_status_table(), q.get_summary()

    def on_refresh():
        """Auto-refresh callback from Timer."""
        q = _batch_queue
        if q is None:
            return [], "无任务", gr.update(active=False), gr.update(choices=[]), gr.update()

        table = q.get_status_table()
        summary = q.get_summary()
        is_busy = q.is_busy()

        # Update download dropdown choices
        completed = q.get_completed_paths()
        choices = [p.name for p in completed]

        return (
            table,
            summary,
            gr.update(active=is_busy),  # stop timer when done
            gr.update(choices=choices),
            gr.update(),  # batch_single_file unchanged
        )

    def on_zip_download():
        """Pack all completed files into a zip."""
        q = _batch_queue
        if q is None:
            gr.Warning("没有可下载的文件")
            return None
        paths = q.get_completed_paths()
        if not paths:
            gr.Warning("没有已完成的音频文件")
            return None
        zip_path = pack_to_zip(paths, output_dir=_BATCH_OUTPUT_BASE)
        if zip_path is None:
            gr.Warning("打包失败")
            return None
        return str(zip_path)

    def on_single_download(filename):
        """Download a single completed file."""
        q = _batch_queue
        if q is None or not filename:
            return None
        paths = q.get_completed_paths()
        for p in paths:
            if p.name == filename:
                return str(p)
        return None

    def on_delete_completed():
        """Delete all completed tasks and their wav files."""
        q = _batch_queue
        if q is None:
            return gr.update(), gr.update(), gr.update()
        count = q.delete_completed()
        if count > 0:
            gr.Info(f"已删除 {count} 个任务及对应文件")
        return q.get_status_table(), q.get_summary(), gr.update(choices=[])

    # ---- Wire up events ----

    batch_start_btn.click(
        fn=on_start_batch,
        inputs=[
            batch_files,
            batch_spk1_audio, batch_spk1_text, batch_spk1_dialect,
            batch_spk2_audio, batch_spk2_text, batch_spk2_dialect,
            batch_seed,
        ],
        outputs=[batch_table, batch_summary, batch_timer, batch_active],
    )

    batch_cancel_btn.click(
        fn=on_cancel,
        outputs=[batch_table, batch_summary],
    )

    batch_timer.tick(
        fn=on_refresh,
        outputs=[
            batch_table,
            batch_summary,
            batch_timer,
            batch_single_select,
            batch_single_file,
        ],
    )

    batch_zip_btn.click(
        fn=on_zip_download,
        outputs=[batch_zip_file],
    )

    batch_single_select.change(
        fn=on_single_download,
        inputs=[batch_single_select],
        outputs=[batch_single_file],
    )

    batch_delete_btn.click(
        fn=on_delete_completed,
        outputs=[batch_table, batch_summary, batch_single_select],
    )
