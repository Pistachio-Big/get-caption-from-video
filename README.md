# get-caption-from-video

Lấy toàn bộ "caption" của một video (Facebook Reel, YouTube, TikTok, Instagram…) ra 1 file `.txt`.

## Chạy

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python caption_pipeline.py "https://www.facebook.com/reel/1735655617492408"
```

Kết quả:
- `output/<video_id>.txt` — báo cáo đầy đủ 4 lớp caption, có timestamp.
- `output/<video_id>.caption.txt` — chỉ caption, gộp thành 1 đoạn văn, bên dưới là bản dịch tiếng Việt.
  Nguồn ưu tiên: phụ đề nền tảng → lời nói (Whisper) → chữ trên hình (OCR) → mô tả bài đăng.
  Dịch qua Google Translate, nếu bị chặn/rate-limit thì tự chuyển sang MyMemory (cả hai miễn phí, cần mạng;
  MyMemory giới hạn ~5000 ký tự/ngày khi không có tài khoản).

Tuỳ chọn: `--model medium|large-v3` (chính xác hơn, chậm hơn), `--lang vi` (ép ngôn ngữ ASR),
`--fps 4` (OCR dày hơn), `--skip-asr`, `--skip-ocr`, `-o file.txt`,
`--translate-to en` (đổi ngôn ngữ đích), `--no-translate`.

Không cần cài ffmpeg riêng (dùng bản đóng gói trong `imageio-ffmpeg`), không cần GPU.

## Pipeline

```
URL ──yt-dlp──► video.mp4 + metadata + phụ đề (nếu có)
                 │            │             │
                 │            │             └─► (2) PLATFORM SUBTITLES  (.vtt/.srt → text)
                 │            └───────────────► (1) POST CAPTION        (description)
                 ├─faster-whisper (VAD)───────► (3) SPEECH TRANSCRIPT   (lời nói, có timestamp)
                 └─ffmpeg fps=2 → RapidOCR ───► (4) ON-SCREEN TEXT      (chữ in trên video, gộp khung trùng)
                                                     ▼
                                           output/<id>.txt
```

"Caption" của một reel có thể là 1 trong 4 thứ trên, nên pipeline lấy cả 4 rồi để người dùng chọn.
Với video mẫu: Facebook không có phụ đề (2), nhưng Whisper (3) và OCR (4) đều cho ra đúng câu thoại;
OCR có lẫn chữ nhãn máy xay "NINJA AUTOiQ" — đó là chữ thật trên hình, không phải phụ đề.

## Khảo sát các project mã nguồn mở

### Tải video / metadata / phụ đề có sẵn
| Project | Ghi chú |
|---|---|
| **yt-dlp** | Chuẩn de-facto, hỗ trợ >1000 site (FB, IG, TikTok, YouTube). Lấy được description + subtitles/auto-captions. **Đã dùng.** |
| gallery-dl | Mạnh với ảnh/post mạng xã hội, video yếu hơn yt-dlp. |
| youtube-transcript-api | Chỉ YouTube, lấy transcript không cần tải video — nhanh nhất nếu nguồn là YouTube. |

### Speech-to-text (ASR)
| Project | Ghi chú |
|---|---|
| openai/whisper | Bản gốc, PyTorch, chậm trên CPU. |
| **faster-whisper** (CTranslate2) | Nhanh hơn ~4x, int8 trên CPU, có VAD (Silero) chống "ảo giác" khi chỉ có nhạc. **Đã dùng.** |
| whisper.cpp | C/C++, rất nhẹ, chạy tốt trên CPU/Apple Silicon, không cần Python. |
| WhisperX | faster-whisper + forced alignment (word-level timestamp) + diarization (ai nói). Hợp khi cần phụ đề karaoke/nhiều người nói. |
| insanely-fast-whisper | HF Transformers + batching, rất nhanh trên GPU. |
| stable-ts | Ổn định timestamp của Whisper, xuất SRT/VTT/ASS đẹp. |
| Vosk / NVIDIA NeMo (Parakeet, Canary) | Vosk: offline, nhẹ, kém chính xác hơn. Parakeet: rất nhanh/chính xác cho tiếng Anh trên GPU. |
| PhoWhisper (VinAI) | Whisper fine-tune cho tiếng Việt — nên dùng nếu video tiếng Việt. |

### Chữ in trên video (burned-in / hard-sub OCR)
| Project | Ghi chú |
|---|---|
| **RapidOCR** | PaddleOCR model chạy qua ONNX Runtime, cài nhẹ, không cần Paddle/Torch. **Đã dùng.** |
| PaddleOCR | Chính xác cao, đa ngôn ngữ (có tiếng Việt), nhưng cài nặng. |
| EasyOCR | Dễ dùng, cần PyTorch. |
| Tesseract | Kinh điển, kém với chữ cách điệu/nền phức tạp của reel. |
| VideOCR / video-subtitle-extractor (YaoFANGUK) | Tool chuyên trích hard-sub: chọn vùng phụ đề, khử trùng lặp khung, xuất SRT. |
| VideoSubFinder | Phát hiện khung có phụ đề rồi mới OCR — hiệu quả với video dài. |

### Hướng nâng cao
- **Vision-LLM** (Qwen2.5-VL, Gemini, Claude…): đưa vài frame vào để đọc chữ cách điệu + lọc chữ không phải phụ đề.
- **Lọc OCR theo vùng**: chỉ giữ hộp chữ ở vùng phụ đề (giữa/dưới khung) hoặc cỡ chữ lớn nhất để bỏ nhãn sản phẩm.
- **Hợp nhất ASR + OCR**: dùng OCR để sửa chính tả tên riêng trong transcript Whisper.
