import os
import json
import shutil
import requests
import logging
import traceback
import imkit as imk
import time
import numpy as np
from datetime import datetime
from typing import List
from PySide6.QtGui import QColor

from modules.detection.processor import TextBlockDetector
from modules.translation.processor import Translator
from modules.utils.textblock import sort_blk_list
from modules.utils.pipeline_utils import inpaint_map, get_config, generate_mask, get_language_code, is_directory_empty
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations
from modules.utils.archives import make
from modules.rendering.render import get_best_render_area, pyside_word_wrap
from modules.utils.device import resolve_device
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer


logger = logging.getLogger(__name__)

# --- INÍCIO DA SOLUÇÃO DEFINITIVA ---
def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Garante que a imagem esteja no formato RGB (3 canais)."""
    if image is not None and image.ndim == 2:  # Se for escala de cinza
        return np.stack([image] * 3, axis=-1)
    return image
# --- FIM DA SOLUÇÃO DEFINITIVA ---


class BatchProcessor:
    """Handles batch processing of comic translation."""
    
    def __init__(
            self, 
            main_page, 
            cache_manager, 
            block_detection_handler, 
            inpainting_handler, 
            ocr_handler
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.block_detection = block_detection_handler
        self.inpainting = inpainting_handler
        self.ocr_handler = ocr_handler

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_images", archive_bname)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        imk.write_image(os.path.join(path, f"{base_name}_translated{extension}"), image)

    def emit_progress(self, index, total, step, steps, change_name):
        """Wrapper around main_page.progress_update.emit that logs a human-readable stage."""
        stage_map = {
            0: 'start-image',
            1: 'text-block-detection',
            2: 'ocr-processing',
            3: 'pre-inpaint-setup',
            4: 'generate-mask',
            5: 'inpainting',
            7: 'translation',
            9: 'text-rendering-prepare',
            10: 'save-and-finish',
        }
        stage_name = stage_map.get(step, f'stage-{step}')
        logger.info(f"Progress: image_index={index}/{total} step={step}/{steps} ({stage_name}) change_name={change_name}")
        self.main_page.progress_update.emit(index, total, step, steps, change_name)

    def log_skipped_image(self, directory, timestamp, image_path, reason="", full_traceback=""):
        skipped_file = os.path.join(directory, f"comic_translate_{timestamp}", "skipped_images.txt")
        with open(skipped_file, 'a', encoding='UTF-8') as file:
            file.write(image_path + "\n")
            file.write(reason + "\n")
            if full_traceback:
                file.write("Full Traceback:\n")
                file.write(full_traceback + "\n")
            file.write("\n")

    def batch_process(self, selected_paths: List[str] = None):
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total_images = len(image_list)

        for index, image_path in enumerate(image_list):

            file_on_display = self.main_page.image_files[self.main_page.curr_img_idx]

            self.emit_progress(index, total_images, 0, 10, True)

            settings_page = self.main_page.settings_page
            source_lang = self.main_page.image_states[image_path]['source_lang']
            target_lang = self.main_page.image_states[image_path]['target_lang']

            target_lang_en = self.main_page.lang_mapping.get(target_lang, None)
            trg_lng_cd = get_language_code(target_lang_en)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            extension = os.path.splitext(image_path)[1]
            directory = os.path.dirname(image_path)

            archive_bname = ""
            for archive in self.main_page.file_handler.archive_info:
                images = archive['extracted_images']
                archive_path = archive['archive_path']
                for img_pth in images:
                    if img_pth == image_path:
                        directory = os.path.dirname(archive_path)
                        archive_bname = os.path.splitext(os.path.basename(archive_path))[0]

            image = imk.read_image(image_path)
            image = ensure_rgb(image)  # <-- APLICA A CORREÇÃO AQUI

            state = self.main_page.image_states.get(image_path, {})
            if state.get('skip', False):
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.log_skipped_image(directory, timestamp, image_path, "User-skipped")
                continue

            self.emit_progress(index, total_images, 1, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            if self.block_detection.block_detector_cache is None:
                self.block_detection.block_detector_cache = TextBlockDetector(self.main_page.settings_page)
            
            blk_list = self.block_detection.block_detector_cache.detect(image)
            
            self.emit_progress(index, total_images, 2, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            if blk_list:
                ocr_model = self.main_page.settings_page.get_tool_selection('ocr')
                cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model)
                self.ocr_handler.ocr.initialize(self.main_page, source_lang)
                try:
                    self.ocr_handler.ocr.process(image, blk_list)
                    self.cache_manager._cache_ocr_results(cache_key, self.main_page.blk_list)
                    source_lang_english = self.main_page.lang_mapping.get(source_lang, source_lang)
                    rtl = True if source_lang_english == 'Japanese' else False
                    blk_list = sort_blk_list(blk_list, rtl)
                except Exception as e:
                    err_msg = str(e.response.json().get("error_description", str(e))) if isinstance(e, requests.exceptions.HTTPError) else str(e)
                    logger.exception(f"OCR processing failed: {err_msg}")
                    reason, full_traceback = f"OCR: {err_msg}", traceback.format_exc()
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "OCR", err_msg)
                    self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                    continue
            else:
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Text Blocks", "")
                self.log_skipped_image(directory, timestamp, image_path, "No text blocks detected")
                continue

            self.emit_progress(index, total_images, 3, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            export_settings = settings_page.get_export_settings()
            if self.inpainting.inpainter_cache is None or self.inpainting.cached_inpainter_key != settings_page.get_tool_selection('inpainter'):
                device = resolve_device(settings_page.is_gpu_enabled())
                inpainter_key = settings_page.get_tool_selection('inpainter')
                InpainterClass = inpaint_map[inpainter_key]
                self.inpainting.inpainter_cache = InpainterClass(device, backend='onnx')
                self.inpainting.cached_inpainter_key = inpainter_key

            config = get_config(settings_page)
            mask = generate_mask(image, blk_list)

            self.emit_progress(index, total_images, 4, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            inpaint_input_img = self.inpainting.inpainter_cache(image, mask, config)
            inpaint_input_img = imk.convert_scale_abs(inpaint_input_img)
            inpaint_input_img = ensure_rgb(inpaint_input_img) # <-- APLICA A CORREÇÃO AQUI TAMBÉM

            patches = self.inpainting.get_inpainted_patches(mask, inpaint_input_img)
            self.main_page.patches_processed.emit(patches, image_path)

            if export_settings['export_inpainted_image']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "cleaned_images", archive_bname)
                if not os.path.exists(path): os.makedirs(path, exist_ok=True)
                imk.write_image(os.path.join(path, f"{base_name}_cleaned{extension}"), inpaint_input_img)

            self.emit_progress(index, total_images, 5, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            extra_context = settings_page.get_llm_settings()['extra_context']
            translator_key = settings_page.get_tool_selection('translator')
            translator = Translator(self.main_page, source_lang, target_lang)
            
            translation_cache_key = self.cache_manager._get_translation_cache_key(image, source_lang, target_lang, translator_key, extra_context)
            
            try:
                translator.translate(blk_list, image, extra_context)
                self.cache_manager._cache_translation_results(translation_cache_key, blk_list)
            except Exception as e:
                err_msg = str(e.response.json().get("error_description", str(e))) if isinstance(e, requests.exceptions.HTTPError) else str(e)
                logger.exception(f"Translation failed: {err_msg}")
                reason, full_traceback = f"Translator: {err_msg}", traceback.format_exc()
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", err_msg)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                continue

            entire_raw_text = get_raw_text(blk_list)
            entire_translated_text = get_raw_translation(blk_list)

            try:
                if (not json.loads(entire_raw_text)) or (not json.loads(entire_translated_text)):
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "Translator", "")
                    self.log_skipped_image(directory, timestamp, image_path, "Translator: empty JSON")
                    continue
            except json.JSONDecodeError as e:
                reason = f"Translator: JSONDecodeError: {str(e)}"
                logger.exception(reason)
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", str(e))
                self.log_skipped_image(directory, timestamp, image_path, reason, traceback.format_exc())
                continue

            if export_settings['export_raw_text']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "raw_texts", archive_bname)
                if not os.path.exists(path): os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, f"{base_name}_raw.txt"), 'w', encoding='UTF-8') as f: f.write(entire_raw_text)

            if export_settings['export_translated_text']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_texts", archive_bname)
                if not os.path.exists(path): os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, f"{base_name}_translated.txt"), 'w', encoding='UTF-8') as f: f.write(entire_translated_text)

            self.emit_progress(index, total_images, 7, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            render_settings = self.main_page.render_settings()
            format_translations(blk_list, trg_lng_cd, upper_case=render_settings.upper_case)
            get_best_render_area(blk_list, image, inpaint_input_img)

            font, font_color = render_settings.font_family, QColor(render_settings.color)
            line_spacing, outline_width = float(render_settings.line_spacing), float(render_settings.outline_width)
            outline_color = QColor(render_settings.outline_color)
            alignment = self.main_page.button_to_alignment[render_settings.alignment_id]
                
            text_items_state = []
            for blk in blk_list:
                if not blk.translation or len(blk.translation) == 1: continue
                translation, font_size = pyside_word_wrap(
                    blk.translation, font, blk.xywh[2], blk.xywh[3], line_spacing, outline_width, 
                    render_settings.bold, render_settings.italic, render_settings.underline, alignment, 
                    render_settings.direction, render_settings.max_font_size, render_settings.min_font_size
                )
                if image_path == file_on_display: self.main_page.blk_rendered.emit(translation, font_size, blk)
                if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']): translation = translation.replace(' ', '')

                text_items_state.append(TextItemProperties(
                    text=translation, font_family=font, font_size=font_size, text_color=font_color,
                    alignment=alignment, line_spacing=line_spacing, outline_color=outline_color,
                    outline_width=outline_width, bold=render_settings.bold, italic=render_settings.italic,
                    underline=render_settings.underline, position=blk.xywh[:2], rotation=blk.angle,
                    scale=1.0, transform_origin=blk.tr_origin_point, width=blk.xywh[2],
                    direction=render_settings.direction, selection_outlines=[
                        OutlineInfo(0, len(translation), outline_color, outline_width, OutlineType.Full_Document)
                    ] if render_settings.outline else [],
                ).to_dict())

            self.main_page.image_states[image_path]['viewer_state']['text_items_state'] = text_items_state
            self.main_page.image_states[image_path]['viewer_state']['push_to_stack'] = True
            
            self.emit_progress(index, total_images, 9, 10, False)
            if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

            self.main_page.image_states[image_path]['blk_list'] = blk_list
            if image_path == file_on_display: self.main_page.blk_list = blk_list
                
            render_save_dir = os.path.join(directory, f"comic_translate_{timestamp}", "translated_images", archive_bname)
            if not os.path.exists(render_save_dir): os.makedirs(render_save_dir, exist_ok=True)
            sv_pth = os.path.join(render_save_dir, f"{base_name}_translated{extension}")

            renderer = ImageSaveRenderer(image)
            viewer_state = self.main_page.image_states[image_path]['viewer_state'].copy()
            patches = self.main_page.image_patches.get(image_path, [])
            renderer.apply_patches(patches)
            renderer.add_state_to_image(viewer_state)
            renderer.save_image(sv_pth)

            self.emit_progress(index, total_images, 10, 10, False)

        archive_info_list = self.main_page.file_handler.archive_info
        if archive_info_list:
            save_as_settings = settings_page.get_export_settings()['save_as']
            for archive_index, archive in enumerate(archive_info_list):
                archive_index_input = total_images + archive_index
                self.emit_progress(archive_index_input, total_images, 1, 3, True)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break

                archive_path = archive['archive_path']
                archive_bname = os.path.splitext(os.path.basename(archive_path))[0]
                archive_directory = os.path.dirname(archive_path)
                save_as_ext = f".{save_as_settings[os.path.splitext(archive_path)[1].lower()]}"
                save_dir = os.path.join(archive_directory, f"comic_translate_{timestamp}", "translated_images", archive_bname)
                check_from = os.path.join(archive_directory, f"comic_translate_{timestamp}")

                self.emit_progress(archive_index_input, total_images, 2, 3, True)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break
                
                make(save_as_ext=save_as_ext, input_dir=save_dir, output_dir=archive_directory, output_base_name=archive_bname)
                
                self.emit_progress(archive_index_input, total_images, 3, 3, True)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled: break
                
                if os.path.exists(save_dir): shutil.rmtree(save_dir)
                if is_directory_empty(check_from): shutil.rmtree(check_from)