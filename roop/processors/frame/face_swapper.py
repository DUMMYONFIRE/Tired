from typing import List, Callable
import cv2
import insightface
import threading
import roop.globals
import roop.processors.frame.core as frame_core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_or_load_face_swapper() -> insightface.FaceAnalysis:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_resources() -> None:
    clear_face_swapper()
    clear_face_reference()


def pre_check_and_download_models() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start_check(source_path: str, target_path: str) -> bool:
    if not is_image(source_path) or not get_one_face(cv2.imread(source_path)):
        update_status('Select a valid image for the source path.', NAME)
        return False

    if not (is_image(target_path) or is_video(target_path)):
        update_status('Select a valid image or video for the target path.', NAME)
        return False

    return True


def post_process_cleanup() -> None:
    clear_resources()


def swap_faces(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_or_load_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        many_faces = get_many_faces(temp_frame) if roop.globals.many_faces else None

        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_faces(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_faces(source_face, target_face, temp_frame)

        cv2.imwrite(temp_frame_path, temp_frame)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = swap_faces(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)

    frame_core.process_video(source_path, temp_frame_paths, process_frames)
