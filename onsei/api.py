# pip3 install fastapi python-multipart uvicorn
# uvicorn onsei.api:app --reload
"""
API to perform an audio comparison and get a graph
"""
import logging
import math
import os
import shutil
import traceback
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, status, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from onsei.pyplot import plot_aligned_pitches_and_phonemes, plot_pitch_and_spectro, plot_pitch_and_phonemes
from onsei.speech_record import SpeechRecord, AlignmentError, NoPhonemeSegmentationError, AlignmentMethod
from onsei.utils import convert_audio

app = FastAPI()


SUPPORTED_FILE_EXTENSIONS = {"wav", "mp3", "ogg"}


def _validate_extension(upload: UploadFile, label: str) -> None:
    extension = upload.filename.split('.')[-1].lower()
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f'{label} {upload.filename} has unsupported extension {extension}, '
                f'should be one of the following: {",".join(SUPPORTED_FILE_EXTENSIONS)}'
            ),
        )


def _write_upload(upload: UploadFile, destination: str) -> None:
    upload.file.seek(0)
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _float_or_none(value: float) -> Optional[float]:
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _serialize_curve(xs, ys) -> List[Dict[str, Optional[float]]]:
    return [
        {"time": float(x), "value": _float_or_none(float(y))}
        for x, y in zip(xs, ys)
    ]


def _serialize_alignment(student_rec: SpeechRecord) -> Dict[str, List[Dict[str, Optional[float]]]]:
    if student_rec.align_ts is None:
        return {}
    alignment_ts = student_rec.ref_rec.align_ts
    ref_pitch = student_rec.ref_rec.norm_aligned_pitch
    student_pitch = student_rec.norm_aligned_pitch
    payload = {
        "reference_pitch": [
            {"time": float(t), "value": _float_or_none(float(v))}
            for t, v in zip(alignment_ts, ref_pitch)
        ],
        "student_pitch": [
            {"time": float(t), "value": _float_or_none(float(v))}
            for t, v in zip(alignment_ts, student_pitch)
        ],
    }
    if student_rec.pitch_diffs_ts and student_rec.pitch_diffs:
        payload["pitch_diff"] = [
            {"time": float(t), "value": _float_or_none(float(v))}
            for t, v in zip(student_rec.pitch_diffs_ts, student_rec.pitch_diffs)
        ]
    return payload


def _serialize_phonemes(phonemes):
    if not phonemes:
        return []
    return [
        {"start": float(start), "end": float(end), "label": label}
        for start, end, label in phonemes
    ]


def _serialize_record(rec: SpeechRecord) -> Dict[str, object]:
    return {
        "name": rec.name,
        "pitch": _serialize_curve(rec.pitch.xs(), rec.pitch_freq),
        "intensity": _serialize_curve(rec.intensity.xs(), rec.intensity.values[0]),
        "phonemes": _serialize_phonemes(rec.phonemes),
        "mean_pitch": _float_or_none(float(rec.mean_pitch_freq)) if rec.mean_pitch_freq is not None else None,
        "std_pitch": _float_or_none(float(rec.std_pitch_freq)) if rec.std_pitch_freq is not None else None,
        "voice_activity": {
            "begin": _float_or_none(float(rec.begin_ts)) if rec.begin_ts is not None else None,
            "end": _float_or_none(float(rec.end_ts)) if rec.end_ts is not None else None,
        },
    }


def _compare_audio_uploads(
    sentence: str,
    align_audios: bool,
    alignment_method: AlignmentMethod,
    fallback_if_no_alignment: bool,
    teacher_audio_file: UploadFile,
    student_audio_file: UploadFile,
):
    for file, label in [(teacher_audio_file, "Reference audio"), (student_audio_file, "Your recording")]:
        _validate_extension(file, label)

    with TemporaryDirectory() as td:
        teacher_audio_filepath = os.path.join(td, teacher_audio_file.filename)
        student_audio_filepath = os.path.join(td, student_audio_file.filename)
        _write_upload(teacher_audio_file, teacher_audio_filepath)
        _write_upload(student_audio_file, student_audio_filepath)

        logging.debug(f"Converting {teacher_audio_filepath} and {student_audio_filepath} to WAV 16KHz mono")

        teacher_wav_filepath = os.path.join(td, "teacher.wav")
        convert_audio(teacher_audio_filepath, teacher_wav_filepath)
        student_wav_filepath = os.path.join(td, "student.wav")
        convert_audio(student_audio_filepath, student_wav_filepath)

        logging.debug(f"Comparing {teacher_wav_filepath} with {student_wav_filepath}")

        mean_distance = None
        teacher_rec = SpeechRecord(teacher_wav_filepath, sentence, name="Teacher")
        student_rec = SpeechRecord(student_wav_filepath, sentence, name="Student")

        try:
            if align_audios:
                student_rec.align_with(teacher_rec, method=alignment_method)
                mean_distance = student_rec.compare_pitch()
        except AlignmentError:
            logging.error(traceback.format_exc())
            if not fallback_if_no_alignment:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Could not align your speech with the reference, try recording again',
                )
        except NoPhonemeSegmentationError as exc:
            logging.error(traceback.format_exc())
            if exc.record == student_rec:
                detail = 'Could not segment the phonemes in your speech, try recording again'
            else:
                detail = 'Could not segment the phonemes in the reference audio'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
        except Exception:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Something went wrong on the server, not your fault :(',
            )

    score = None
    if mean_distance is not None:
        score = int(1.0 / (mean_distance + 1.0) * 100)

    return teacher_rec, student_rec, score, mean_distance


@app.post("/compare/graph.png")
def post_compare_graph_png(
    sentence: str = Form(...),
    align_audios: bool = Form(True),
    show_all_graphs: bool = Form(False),
    alignment_method: AlignmentMethod = Form(AlignmentMethod.phonemes),
    fallback_if_no_alignment: bool = Form(True),
    teacher_audio_file: UploadFile = File(...),
    student_audio_file: UploadFile = File(...),
):
    if not show_all_graphs and not align_audios:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Can't have both show_all_graphs and align_audios set to false !")

    teacher_rec, student_rec, score, mean_distance = _compare_audio_uploads(
        sentence=sentence,
        align_audios=align_audios,
        alignment_method=alignment_method,
        fallback_if_no_alignment=fallback_if_no_alignment,
        teacher_audio_file=teacher_audio_file,
        student_audio_file=student_audio_file,
    )

    show_alignment = align_audios and score is not None
    if show_alignment:
        nb_graphs = 1 + int(show_all_graphs) * 2
    else:
        nb_graphs = 2

    plt.figure(figsize=(12, max(nb_graphs * 2, 4)))
    idx = 1
    if show_alignment:
        plt.subplot(nb_graphs * 100 + 10 + idx)
        plt.title(f"Similarity score: {score}%")
        plot_aligned_pitches_and_phonemes(student_rec)
        idx += 1
    if not show_alignment or show_all_graphs:
        plt.subplot(nb_graphs * 100 + 10 + idx)
        plot_pitch_and_phonemes(student_rec, 'r', "Your recording")
        if not align_audios:
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower right", ncol=1)
        idx += 1
        plt.subplot(nb_graphs * 100 + 10 + idx)
        plot_pitch_and_phonemes(teacher_rec, 'b', "Reference audio")
        if not align_audios:
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower right", ncol=1)
        idx += 1

    b = BytesIO()
    plt.savefig(b, format='png')
    b.seek(0)

    return StreamingResponse(b, media_type="image/png")


@app.post("/compare/data")
def post_compare_data(
    sentence: str = Form(...),
    align_audios: bool = Form(True),
    show_all_graphs: bool = Form(False),
    alignment_method: AlignmentMethod = Form(AlignmentMethod.phonemes),
    fallback_if_no_alignment: bool = Form(True),
    teacher_audio_file: UploadFile = File(...),
    student_audio_file: UploadFile = File(...),
):
    if not show_all_graphs and not align_audios:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can't have both show_all_graphs and align_audios set to false !",
        )

    teacher_rec, student_rec, score, mean_distance = _compare_audio_uploads(
        sentence=sentence,
        align_audios=align_audios,
        alignment_method=alignment_method,
        fallback_if_no_alignment=fallback_if_no_alignment,
        teacher_audio_file=teacher_audio_file,
        student_audio_file=student_audio_file,
    )

    response: Dict[str, object] = {
        "score": score,
        "mean_distance": _float_or_none(mean_distance) if mean_distance is not None else None,
        "alignment_method": alignment_method.value if alignment_method else None,
        "graphs": [],
    }

    if align_audios and score is not None:
        response["graphs"].append(
            {
                "type": "aligned_pitch",
                "data": _serialize_alignment(student_rec),
            }
        )

    if (not align_audios) or show_all_graphs:
        response["graphs"].append(
            {
                "type": "student_pitch",
                "data": _serialize_record(student_rec),
            }
        )
        response["graphs"].append(
            {
                "type": "reference_pitch",
                "data": _serialize_record(teacher_rec),
            }
        )

    return response


@app.post("/graph.png")
def post_graph_png(
    sentence: str = Form(...),
    audio_file: UploadFile = File(...),
):
    file = audio_file
    _validate_extension(file, "Audio file")

    with TemporaryDirectory() as td:
        audio_filepath = os.path.join(td, audio_file.filename)
        _write_upload(audio_file, audio_filepath)

        logging.debug(f"Converting {audio_filepath} to WAV 16KHz mono")

        wav_filepath = os.path.join(td, "audio.wav")
        convert_audio(audio_filepath, wav_filepath)

        try:
            rec = SpeechRecord(wav_filepath, sentence, name="Reference")
        except NoPhonemeSegmentationError as exc:
            logging.error(traceback.format_exc())
            detail = f'Could not segment the phonemes in the audio'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=detail)
        except Exception:
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f'Something went wrong on the server, not your fault :(')

        plt.figure(figsize=(12, 6))
        plot_pitch_and_phonemes(rec, 'b', "Reference audio")

    b = BytesIO()
    plt.savefig(b, format='png')
    b.seek(0)

    return StreamingResponse(b, media_type="image/png")


@app.post("/graph/data")
def post_graph_data(
    sentence: str = Form(...),
    audio_file: UploadFile = File(...),
):
    _validate_extension(audio_file, "Audio file")

    with TemporaryDirectory() as td:
        audio_filepath = os.path.join(td, audio_file.filename)
        _write_upload(audio_file, audio_filepath)

        logging.debug(f"Converting {audio_filepath} to WAV 16KHz mono")

        wav_filepath = os.path.join(td, "audio.wav")
        convert_audio(audio_filepath, wav_filepath)

        try:
            rec = SpeechRecord(wav_filepath, sentence, name="Reference")
        except NoPhonemeSegmentationError:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Could not segment the phonemes in the audio',
            )
        except Exception:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Something went wrong on the server, not your fault :(',
            )

    return {
        "record": _serialize_record(rec),
    }


@app.get("/")
async def get_root():
    """ Form for testing """
    content = """
<body>
<form action="/compare/graph.png" enctype="multipart/form-data" method="post">
Teacher audio file: <input name="teacher_audio_file" type="file"></br>
Student audio file: <input name="student_audio_file" type="file"></br>
Sentence: <input name="sentence" type="text"></br>
</br>
<input name="align_audios" type="checkbox" checked>Align audios ?</br>
<input name="show_all_graphs" type="checkbox">Show all graphs ?</br>
Align speech using: <select name="alignment_method">
  <option value="phonemes">Phonemes</option>
  <option value="intensity">Intensity</option>
</select></br>
</br>
<input type="submit">
</form>
</br>
<a href="https://github.com/itsupera/onsei#readme">What is this ?</a></br>
</body>
    """
    return HTMLResponse(content=content)
