from __future__ import annotations

import os
import imkit as imk
from PIL import Image
import numpy as np
from typing import TYPE_CHECKING, List
from PySide6 import QtCore, QtWidgets, QtGui

from app.ui.dayu_widgets.clickable_card import ClickMeta
from app.ui.dayu_widgets.message import MMessage
from app.ui.commands.image import SetImageCommand
from app.ui.commands.inpaint import PatchInsertCommand
from app.ui.commands.inpaint import PatchCommandBase
from app.ui.commands.box import AddTextItemCommand
from app.ui.list_view_image_loader import ListViewImageLoader

if TYPE_CHECKING:
    from controller import ComicTranslate


class ImageStateController:
    def __init__(self, main: ComicTranslate):
        self.main = main
        
        self.page_list_loader = ListViewImageLoader(
            self.main.page_list,
            avatar_size=(35, 50)
        )

    def load_initial_image(self, file_paths: List[str]):
        file_paths = self.main.file_handler.prepare_files(file_paths)
        self.main.image_files = file_paths

        if file_paths:
            return self.load_image(file_paths[0])
        return None
    
    def load_image(self, file_path: str) -> np.ndarray:
        # --- INÍCIO DA CORREÇÃO DEFINITIVA ---
        # Esta função agora garante que qualquer imagem retornada seja RGB.
        
        def _load_and_convert(path):
            img = imk.read_image(path)
            if img is not None and img.ndim == 2:
                # Se a imagem for escala de cinza, converte para RGB
                return np.stack([img] * 3, axis=-1)
            return img

        if file_path in self.main.image_data and self.main.image_data[file_path] is not None:
            return self.main.image_data[file_path]

        if file_path in self.main.image_history:
            current_index = self.main.current_history_index[file_path]
            current_temp_path = self.main.image_history[file_path][current_index]
            rgb_image = _load_and_convert(current_temp_path)
            if rgb_image is not None:
                return rgb_image

        # Se tudo falhar, carrega do arquivo original e converte
        rgb_image = _load_and_convert(file_path)
        return rgb_image
        # --- FIM DA CORREÇÃO DEFINITIVA ---

    def clear_state(self):
        self.main.image_files = []
        self.main.image_states.clear()
        self.main.image_data.clear()
        self.main.image_history.clear()
        self.main.current_history_index.clear()
        self.main.blk_list = []
        self.main.displayed_images.clear()
        self.main.image_viewer.clear_rectangles(page_switch=True)
        self.main.image_viewer.clear_brush_strokes(page_switch=True)
        self.main.s_text_edit.clear()
        self.main.t_text_edit.clear()
        self.main.image_viewer.clear_text_items()
        self.main.loaded_images = []
        self.main.in_memory_history.clear()
        self.main.undo_stacks.clear()
        self.main.image_patches.clear()
        self.main.in_memory_patches.clear()
        self.main.project_file = None
        self.main.image_cards.clear()
        self.main.current_card = None
        self.main.page_list.blockSignals(True)
        self.main.page_list.clear()
        self.main.page_list.blockSignals(False)
        self.page_list_loader.clear()
        self.main.curr_img_idx = -1

    def thread_load_images(self, paths: List[str]):
        if paths and paths[0].lower().endswith('.ctpr'):
            self.main.project_ctrl.thread_load_project(paths[0])
            return
        self.clear_state()
        self.main.run_threaded(self.load_initial_image, self.on_initial_image_loaded, self.main.default_error_handler, None, paths)

    def thread_insert(self, paths: List[str]):
        if self.main.image_files:
            def on_files_prepared(prepared_files):
                self.save_current_image_state()
                insert_position = len(self.main.image_files)
                for i, file_path in enumerate(prepared_files):
                    self.main.image_files.insert(insert_position + i, file_path)
                    self.main.image_data[file_path] = None
                    self.main.image_history[file_path] = [file_path]
                    self.main.in_memory_history[file_path] = []
                    self.main.current_history_index[file_path] = 0
                    self.save_image_state(file_path)
                    stack = QtGui.QUndoStack(self.main)
                    self.main.undo_stacks[file_path] = stack
                    self.main.undo_group.addStack(stack)
                if self.main.webtoon_mode:
                    if self.main.image_viewer.webtoon_manager.insert_pages(prepared_files, insert_position):
                        self.update_image_cards()
                        self.main.page_list.blockSignals(True)
                        self.main.page_list.setCurrentRow(insert_position)
                        self.highlight_card(insert_position)
                        self.main.page_list.blockSignals(False)
                        self.main.curr_img_idx = insert_position
                    else:
                        current_page = max(0, self.main.curr_img_idx)
                        self.main.image_viewer.webtoon_manager.load_images_lazy(self.main.image_files, current_page)
                        self.update_image_cards()
                        self.main.page_list.blockSignals(True)
                        self.main.page_list.setCurrentRow(current_page)
                        self.highlight_card(current_page)
                        self.main.page_list.blockSignals(False)
                else:
                    self.update_image_cards()
                    self.main.page_list.setCurrentRow(insert_position)
                    path = prepared_files[0]
                    new_index = self.main.image_files.index(path)
                    im = self.load_image(path)
                    self.display_image_from_loaded(im, new_index, False)
            self.main.run_threaded(lambda: self.main.file_handler.prepare_files(paths, True), on_files_prepared, self.main.default_error_handler)
        else:
            self.thread_load_images(paths)

    def on_initial_image_loaded(self, rgb_image: np.ndarray):
        if rgb_image is not None:
            self.main.image_data[self.main.image_files[0]] = rgb_image
            self.main.image_history[self.main.image_files[0]] = [self.main.image_files[0]]
            self.main.in_memory_history[self.main.image_files[0]] = [rgb_image.copy()]
            self.main.current_history_index[self.main.image_files[0]] = 0
            self.save_image_state(self.main.image_files[0])
        for file in self.main.image_files:
            self.save_image_state(file)
            stack = QtGui.QUndoStack(self.main)
            self.main.undo_stacks[file] = stack
            self.main.undo_group.addStack(stack)
        if self.main.image_files:
            self.main.page_list.blockSignals(True)
            self.update_image_cards()
            self.main.page_list.blockSignals(False)
            self.main.page_list.setCurrentRow(0)
            self.main.loaded_images.append(self.main.image_files[0])
        else:
            self.main.image_viewer.clear_scene()
        self.main.image_viewer.resetTransform()
        self.main.image_viewer.fitInView()

    def update_image_cards(self):
        self.main.page_list.clear()
        self.main.image_cards.clear()
        self.main.current_card = None
        for file_path in self.main.image_files:
            file_name = os.path.basename(file_path)
            list_item = QtWidgets.QListWidgetItem(file_name)
            card = ClickMeta(extra=False, avatar_size=(35, 50))
            card.setup_data({"title": file_name})
            list_item.setSizeHint(card.sizeHint())
            if self.main.image_states.get(file_path, {}).get('skip'):
                font = list_item.font()
                font.setStrikeOut(True)
                list_item.setFont(font)
                card.set_skipped(True)
            self.main.page_list.addItem(list_item)
            self.main.page_list.setItemWidget(list_item, card)
            self.main.image_cards.append(card)
        self.page_list_loader.set_file_paths(self.main.image_files, self.main.image_cards)

    def on_card_selected(self, current, previous):
        if current:
            index = self.main.page_list.row(current)
            self.main.curr_tblock_item = None
            self.page_list_loader.force_load_image(index)
            if getattr(self.main, '_processing_page_change', False): return
            if self.main.webtoon_mode:
                if self.main.image_viewer.hasPhoto():
                    self.main.curr_img_idx = index
                    self.main.image_viewer.scroll_to_page(index)
                    file_path = self.main.image_files[index]
                    if file_path in self.main.image_states:
                        state = self.main.image_states[file_path]
                        self.main.s_combo.setCurrentText(state.get('source_lang', ''))
                        self.main.t_combo.setCurrentText(state.get('target_lang', ''))
                    self.main.text_ctrl.clear_text_edits()
                else:
                    self.main.run_threaded(lambda: self.load_image(self.main.image_files[index]), lambda result: self.display_image_from_loaded(result, index), self.main.default_error_handler)
            else:
                self.main.run_threaded(lambda: self.load_image(self.main.image_files[index]), lambda result: self.display_image_from_loaded(result, index), self.main.default_error_handler)

    def navigate_images(self, direction: int):
        if self.main.image_files:
            new_index = self.main.curr_img_idx + direction
            if 0 <= new_index < len(self.main.image_files):
                self.main.page_list.setCurrentRow(new_index)

    def highlight_card(self, index: int):
        for card in self.main.image_cards: card.set_highlight(False)
        if 0 <= index < len(self.main.image_cards):
            self.main.image_cards[index].set_highlight(True)
            self.main.current_card = self.main.image_cards[index]
        else:
            self.main.current_card = None

    def on_selection_changed(self, selected_indices: list):
        for i, card in enumerate(self.main.image_cards): card.set_highlight(i in selected_indices)
        if selected_indices:
            current_index = selected_indices[-1]
            if 0 <= current_index < len(self.main.image_cards): self.main.current_card = self.main.image_cards[current_index]
        else:
            self.main.current_card = None

    def handle_image_deletion(self, file_names: list[str]):
        self.save_current_image_state()
        for file_name in file_names:
            file_path = next((f for f in self.main.image_files if os.path.basename(f) == file_name), None)
            if file_path:
                self.main.image_files.remove(file_path)
                for d in [self.main.image_data, self.main.image_history, self.main.in_memory_history, self.main.current_history_index, self.main.image_states, self.main.image_patches, self.main.in_memory_patches, self.main.undo_stacks]:
                    d.pop(file_path, None)
                if file_path in self.main.undo_stacks: self.main.undo_group.removeStack(self.main.undo_stacks[file_path])
                self.main.displayed_images.discard(file_path)
                if file_path in self.main.loaded_images: self.main.loaded_images.remove(file_path)
        if self.main.webtoon_mode:
            if self.main.image_files:
                deleted_file_paths = [fp for fp in self.main.image_viewer.webtoon_manager.image_loader.image_file_paths if os.path.basename(fp) in file_names]
                if self.main.image_viewer.webtoon_manager.remove_pages(deleted_file_paths):
                    if self.main.curr_img_idx >= len(self.main.image_files): self.main.curr_img_idx = len(self.main.image_files) - 1
                    current_page = max(0, self.main.curr_img_idx)
                else:
                    current_page = max(0, self.main.curr_img_idx)
                    self.main.image_viewer.webtoon_manager.load_images_lazy(self.main.image_files, current_page)
                self.update_image_cards()
                self.main.page_list.blockSignals(True)
                self.main.page_list.setCurrentRow(current_page)
                self.highlight_card(current_page)
                self.main.page_list.blockSignals(False)
            else:
                self.main.webtoon_mode = False
                self.main.image_viewer.webtoon_manager.clear()
                self.main.curr_img_idx = -1
                self.main.central_stack.setCurrentWidget(self.main.drag_browser)
                self.update_image_cards()
        else:
            if self.main.image_files:
                if self.main.curr_img_idx >= len(self.main.image_files): self.main.curr_img_idx = len(self.main.image_files) - 1
                new_index = max(0, self.main.curr_img_idx - 1)
                file = self.main.image_files[new_index]
                im = self.load_image(file)
                self.display_image_from_loaded(im, new_index, False)
                self.update_image_cards()
                self.main.page_list.blockSignals(True)
                self.main.page_list.setCurrentRow(new_index)
                self.highlight_card(new_index)
                self.main.page_list.blockSignals(False)
            else:
                self.main.curr_img_idx = -1
                self.main.central_stack.setCurrentWidget(self.main.drag_browser)
                self.update_image_cards()

    def handle_toggle_skip_images(self, file_names: list[str], skip_status: bool):
        for name in file_names:
            path = next((p for p in self.main.image_files if os.path.basename(p) == name), None)
            if not path: continue
            self.main.image_states.get(path, {})['skip'] = skip_status
            idx = self.main.image_files.index(path)
            item = self.main.page_list.item(idx)
            fnt = item.font()
            fnt.setStrikeOut(skip_status)
            item.setFont(fnt)
            card = self.main.page_list.itemWidget(item)
            if card: card.set_skipped(skip_status)

    def display_image_from_loaded(self, rgb_image, index: int, switch_page: bool = True):
        if rgb_image is None: return
        file_path = self.main.image_files[index]
        self.main.image_data[file_path] = rgb_image
        if file_path not in self.main.image_history:
            self.main.image_history[file_path] = [file_path]
            self.main.in_memory_history[file_path] = [rgb_image.copy()]
            self.main.current_history_index[file_path] = 0
        self.display_image(index, switch_page)
        if file_path not in self.main.loaded_images:
            self.main.loaded_images.append(file_path)
            if len(self.main.loaded_images) > self.main.max_images_in_memory:
                oldest_image = self.main.loaded_images.pop(0)
                del self.main.image_data[oldest_image]
                self.main.in_memory_history[oldest_image] = []
                self.main.in_memory_patches.pop(oldest_image, None)

    def set_image(self, rgb_img: np.ndarray, push: bool = True):
        if self.main.curr_img_idx >= 0:
            file_path = self.main.image_files[self.main.curr_img_idx]
            command = SetImageCommand(self.main, file_path, rgb_img)
            if push:
                self.main.undo_group.activeStack().push(command)
            else:
                command.redo()

    def load_patch_state(self, file_path: str):
        mem_list = self.main.in_memory_patches.setdefault(file_path, [])
        for saved in self.main.image_patches.get(file_path, []):
            match = next((m for m in mem_list if m['hash'] == saved['hash']), None)
            if match:
                prop = {'bbox': saved['bbox'], 'image': match['image'], 'hash': saved['hash']}
            else:
                rgb_img = imk.read_image(saved['png_path'])
                prop = {'bbox': saved['bbox'], 'image': rgb_img, 'hash': saved['hash']}
                self.main.in_memory_patches[file_path].append(prop)
            if not PatchCommandBase.find_matching_item(self.main.image_viewer._scene, prop):   
                PatchCommandBase.create_patch_item(prop, self.main.image_viewer)

    def save_current_image(self, file_path: str):
        if self.main.webtoon_mode:
            final_rgb, _ = self.main.image_viewer.get_visible_area_image(paint_all=True)
        else:
            final_rgb = self.main.image_viewer.get_image_array(paint_all=True)
        pil_img = Image.fromarray(final_rgb)
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        jpeg_quality = settings.value('export/jpeg_quality', 95, type=int)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg']:
            pil_img.save(file_path, quality=jpeg_quality, optimize=True)
        else:
            pil_img.save(file_path)

    def save_image_state(self, file: str):
        skip_status = self.main.image_states.get(file, {}).get('skip', False)
        self.main.image_states[file] = {
            'viewer_state': self.main.image_viewer.save_state(),
            'source_lang': self.main.s_combo.currentText(),
            'target_lang': self.main.t_combo.currentText(),
            'brush_strokes': self.main.image_viewer.save_brush_strokes(),
            'blk_list': self.main.blk_list.copy(),
            'skip': skip_status,
        }

    def save_current_image_state(self):
        if self.main.curr_img_idx >= 0:
            current_file = self.main.image_files[self.main.curr_img_idx]
            self.save_image_state(current_file)

    def load_image_state(self, file_path: str):
        rgb_image = self.main.image_data.get(file_path)
        if rgb_image is None: return
        self.set_image(rgb_image, push=False)
        if file_path in self.main.image_states:
            state = self.main.image_states[file_path]
            self.main.blk_list = state['blk_list'].copy()
            self.main.image_viewer.load_state(state['viewer_state'])
            self.main.s_combo.setCurrentText(state['source_lang'])
            self.main.t_combo.setCurrentText(state['target_lang'])
            self.main.image_viewer.load_brush_strokes(state['brush_strokes'])
            if state.get('viewer_state', {}).get('push_to_stack', False):
                self.main.undo_stacks[file_path].beginMacro('text_items_rendered')
                for text_item in self.main.image_viewer.text_items:
                    self.main.text_ctrl.connect_text_item_signals(text_item)
                    command = AddTextItemCommand(self.main, text_item)
                    self.main.undo_stacks[file_path].push(command)
                self.main.undo_stacks[file_path].endMacro()
                state['viewer_state']['push_to_stack'] = False
            else:
                for text_item in self.main.image_viewer.text_items:
                    self.main.text_ctrl.connect_text_item_signals(text_item)
            for rect_item in self.main.image_viewer.rectangles:
                self.main.rect_item_ctrl.connect_rect_item_signals(rect_item)
            self.load_patch_state(file_path)
        self.main.text_ctrl.clear_text_edits()

    def display_image(self, index: int, switch_page: bool = True):
        if 0 <= index < len(self.main.image_files):
            if switch_page: self.save_current_image_state()
            self.main.curr_img_idx = index
            file_path = self.main.image_files[index]
            if file_path in self.main.undo_stacks:
                self.main.undo_group.setActiveStack(self.main.undo_stacks[file_path])
            
            first_time_display = file_path not in self.main.displayed_images
            self.load_image_state(file_path)
            
            if self.main.webtoon_mode:
                self.main.image_viewer.scroll_to_page(index)
                self.main.central_stack.setCurrentWidget(self.main.image_viewer)
            else:
                self.main.central_stack.setCurrentWidget(self.main.image_viewer)
                
            self.main.central_stack.layout().activate()
            
            if first_time_display and not self.main.webtoon_mode:
                self.main.image_viewer.fitInView()
                self.main.displayed_images.add(file_path)

    def on_image_processed(self, index: int, image: np.ndarray, image_path: str):
        file_on_display = self.main.image_files[self.main.curr_img_idx]
        current_batch_file = self.main.selected_batch[index] if self.main.selected_batch else self.main.image_files[index]
        if current_batch_file == file_on_display:
            self.set_image(image)
        else:
            command = SetImageCommand(self.main, image_path, image, False)
            self.main.undo_stacks[current_batch_file].push(command)
            self.main.image_data[image_path] = image

    def on_image_skipped(self, image_path: str, skip_reason: str, error: str):
        message = { 
            "Text Blocks": f"{QtCore.QCoreApplication.translate('Messages', 'No Text Blocks Detected.\nSkipping:')} {image_path}\n{error}", 
            "OCR": f"{QtCore.QCoreApplication.translate('Messages', 'Could not OCR detected text.\nSkipping:')} {image_path}\n{error}",
            "Translator": f"{QtCore.QCoreApplication.translate('Messages', 'Could not get translations.\nSkipping:')} {image_path}\n{error}"
        }
        text = message.get(skip_reason, f"Unknown skip reason: {skip_reason}. Error: {error}")
        MMessage.info(text=text, parent=self.main, duration=5, closable=True)

    def on_inpaint_patches_processed(self, patches: list, file_path: str):
        target_stack = self.main.undo_stacks[file_path]
        should_display = False
        if self.main.webtoon_mode:
            page_index = self.main.image_files.index(file_path) if file_path in self.main.image_files else None
            if page_index is not None and page_index in self.main.image_viewer.webtoon_manager.loaded_pages:
                should_display = True
        else:
            file_on_display = self.main.image_files[self.main.curr_img_idx] if self.main.image_files else None
            should_display = (file_path == file_on_display)
        command = PatchInsertCommand(self.main, patches, file_path, display=should_display)
        target_stack.push(command)

    def apply_inpaint_patches(self, patches):
        command = PatchInsertCommand(self.main, patches, self.main.image_files[self.main.curr_img_idx])
        self.main.undo_group.activeStack().push(command)

    def cleanup(self):
        if hasattr(self, 'page_list_loader'):
            self.page_list_loader.shutdown()