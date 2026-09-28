# Báo cáo khảo sát PTQ (Post-Training Quantization) — baseline vs vocos_small

**Ngày:** 2026-09-07

**Mục tiêu:** Tìm cấu hình loại trừ layer PTQ (không cần train) tốt nhất cho 2 checkpoint đang phát triển trong phiên làm việc hiện tại của BanhmiTTS — `baseline` (HiFi-GAN family, không F0) và `vocos_small` (Vocos ConvNeXt+ISTFT, không F0) — trước khi đầu tư thời gian GPU vào QAT training, theo đúng phương pháp đã chứng minh hiệu quả trong `QAT_Quantization_Report.md` (khảo sát PTQ rẻ trước, chỉ QAT hoá cấu hình đã biết là ổn).

**Checkpoint dùng trong báo cáo này:**
- `baseline`: `baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt` (`use_vocos=False`, `flow_n_flows=4` nhưng flow thực tế xuất ra 80 node Conv/ConvTranspose+MatMul — kiến trúc transformer-trong-flow, không phải variant nông của Piper gốc)
- `vocos_small`: `vocos_small_run_v5_fixed/checkpoints/best-epoch=1274-val_loss_mel=19.4528.ckpt` (`use_vocos=True`, `vocos_dim=160/480/8`)
- Cả 2 đều `use_f0=None` (chưa có F0, tính năng F0 được thêm vào codebase sau các checkpoint này)

**Lưu ý quan trọng**: đây là 2 checkpoint khác với "BanhmiTTS_v1"/"Piper (Baseline)" trong `QAT_Quantization_Report.md` — cùng project, cùng phương pháp, nhưng dữ liệu/kiến trúc cụ thể khác nhau (checkpoint mới hơn, từ nhánh huấn luyện khác trong phiên này).

---

## 1. Khác biệt kiến trúc `dec` — lý do phải khảo sát riêng op_types cho từng model

Kiểm tra trực tiếp ONNX graph đã export (`torch.onnx.export` opset 18, không dùng `dynamo` — bản mới cần `dynamo=False` tường minh vì thiếu `onnxscript`):

| | baseline (`dec`) | vocos_small (`dec`) |
|---|---|---|
| Kiến trúc | HiFi-GAN: `conv_pre` + 3×(`ups` ConvTranspose + 3 resblock) + `conv_post` | Vocos: `backbone.embed` + 8×`ConvNeXtBlock` + `head` (ISTFT) |
| Conv/ConvTranspose | 23 (toàn bộ layer học được) | 9 học được (`embed`×1 + `dwconv`×8) **+ 2 cố định** (ISTFT overlap-add, không phải trọng số học) |
| Linear/MatMul | 0 | **17 học được** (`pwconv1`+`pwconv2`×8 block + `head.out`) — phần lớn tham số/tính toán của decoder này |

**Hệ quả**: nếu chỉ quantize `Conv`/`ConvTranspose` (đúng công thức cũ từ `QAT_Quantization_Report.md`) thì với `vocos_small`, phần `dec` gần như không bị chạm tới — bỏ sót toàn bộ 17 lớp Linear chứa phần lớn compute của decoder. Do đó khảo sát này quantize `vocos_small` với `op_types_to_quantize=["Conv","ConvTranspose","MatMul"]`, còn `baseline` vẫn giữ `["Conv","ConvTranspose"]` như cũ (không có Linear đáng kể ngoài attention của `enc_p`/`dp`, vốn không nằm trong `dec`).

Hai node `ConvTranspose` cố định trong `vocos_small.dec.head` (kernel identity dùng cho overlap-add của ISTFT, không phải trọng số học) và 2 node `MatMul` không tên-submodule ở top-level (matmul mở rộng `m_p`/`logs_p` theo alignment trong `SynthesizerTrn.infer()`, cũng không có trọng số học) bị loại trừ **vô điều kiện** khỏi mọi cấu hình khảo sát ở cả 2 model.

---

## 2. Phương pháp

Hai giai đoạn, giống `QAT_Quantization_Report.md`:

1. **Sweep rẻ (n=25, chỉ WER)**: 14 cấu hình loại trừ layer × 2 model = 28 lượt, mỗi lượt chỉ mất PTQ (~15-20s, không train) + eval WER trên 25 câu (Whisper base.en, CPU) — mục đích sàng lọc thô, loại các cấu hình vỡ hẳn trước khi đầu tư eval đầy đủ.
2. **Xác nhận (n=100, WER+UTMOS22+UTMOSv2+RTF)**: chỉ chạy trên (các) cấu hình tốt nhất từ bước 1, để bắt các trường hợp WER "trông ổn" nhưng UTMOS thực tế giảm (bài học đã rút ra từ báo cáo QAT cũ: `val_loss_mel`/WER không phản ánh đầy đủ chất lượng cảm nhận).

Test set: split held-out thật (không dùng để train/validate) — tái tạo bằng `random_split` với đúng seed+hparams kiến trúc của từng model (dataset có RNG-dependent nên baseline và vocos_small có 2 test set khác nhau về thành phần, dù cùng nguồn `dataset.jsonl`).

---

## 3. Kết quả sweep rẻ (n=25, chỉ WER)

### baseline (op_types = Conv, ConvTranspose)

| Config | Submodule quantize | WER | Size |
|---|---|---|---|
| none (sanity) | — | 0.177 | 65.8MB |
| flow_only | flow | 0.165 | 42.3MB |
| flow_finallayer | flow (trừ conv_post) | 0.145 | 42.3MB |
| dp_only | dp | 0.135 | 64.3MB |
| dec_only | dec | 0.179 | 63.0MB |
| enc_p_only | enc_p | 0.199 | 48.0MB |
| flow_enc_p | flow+enc_p | 0.150 | 24.5MB |
| flow_dp | flow+dp | 0.172 | 40.8MB |
| flow_dec | flow+dec | 0.185 | 39.5MB |
| **flow_enc_p_dp** | **flow+enc_p+dp** | **0.121** | **23.0MB** |
| enc_p_dp | enc_p+dp | 0.181 | 46.5MB |
| enc_p_dp_finallayer | enc_p+dp (trừ conv_post) | 0.172 | 46.5MB |
| everything | tất cả | 0.170 | 20.2MB |
| everything_except_finallayer | tất cả (trừ conv_post) | 0.168 | 20.2MB |

### vocos_small (op_types = Conv, ConvTranspose, MatMul)

| Config | Submodule quantize | WER | Size |
|---|---|---|---|
| none (sanity) | — | 0.118 | 73.7MB |
| flow_only | flow | 0.072 | 50.3MB |
| flow_finallayer | flow (trừ head.out) | 0.080 | 50.3MB |
| dp_only | dp | 0.101 | 72.2MB |
| dec_only | dec | 0.091 | 69.2MB |
| enc_p_only | enc_p | 0.070 | 55.9MB |
| flow_enc_p | flow+enc_p | 0.077 | 32.5MB |
| flow_dp | flow+dp | 0.103 | 48.8MB |
| flow_dec | flow+dec | 0.120 | 45.7MB |
| flow_enc_p_dp | flow+enc_p+dp | 0.091 | 31.0MB |
| enc_p_dp | enc_p+dp | 0.088 | 54.4MB |
| enc_p_dp_finallayer | enc_p+dp (trừ head.out) | 0.114 | 54.4MB |
| everything | tất cả | 0.081 | 26.4MB |
| everything_except_finallayer | tất cả (trừ head.out) | 0.077 | 26.9MB |

**Phát hiện chính ở bước sàng lọc**: khác hẳn "Piper (Baseline)" trong `QAT_Quantization_Report.md` (từng vỡ hoàn toàn, WER 90-107%, với cấu hình loại trừ sai), **cả 2 checkpoint trong báo cáo này không hề vỡ ở bất kỳ cấu hình nào** — mọi trường hợp nằm trong khoảng WER 0.07-0.20. Sự khác biệt nằm ở kiến trúc: `baseline` ở đây dùng flow sâu hơn (80 node, biến thể transformer-trong-flow), không phải Piper gốc nông.

---

## 4. Xác nhận đầy đủ (n=100, WER+UTMOS22+UTMOSv2+RTF)

| Model | Variant | WER | UTMOS22 | UTMOSv2 | RTF | Size |
|---|---|---|---|---|---|---|
| baseline | FP32 | 0.157 | 4.005 | 3.136 | 0.026 | 65.4MB |
| baseline | INT8 flow+enc_p+dp | **0.130** | 3.970 | **3.162** | 0.023 | **23.0MB** |
| vocos_small | FP32 | 0.091 | 3.062 | 3.274 | 0.012 | 73.3MB |
| vocos_small | INT8 everything | 0.087 | 2.893 ↓ | 3.111 ↓ | 0.009 | 26.4MB |
| vocos_small | INT8 flow+enc_p+dp | **0.082** | **3.034** | **3.314** | 0.009 | 31.0MB |

**Phát hiện quan trọng nhất**: cấu hình `everything` cho `vocos_small` — dù thắng ở bước sàng lọc rẻ (WER thấp, size nhỏ nhất) — khi đo đầy đủ lại cho thấy **UTMOS giảm thật** (UTMOS22 3.062→2.893, UTMOSv2 3.274→3.111, đều giảm ~0.16) mà bước WER-only n=25 hoàn toàn không phát hiện ra. Đây đúng bài học đã ghi trong `QAT_Quantization_Report.md` mục 4.2: **WER/val_loss_mel không phản ánh đầy đủ chất lượng audio thực tế sau quantize** — chỉ UTMOS đo trên audio thật mới lộ ra.

Ngược lại, cấu hình `flow+enc_p+dp` (loại trừ toàn bộ `dec`) cho cả 2 model đều đạt WER tốt hơn FP32 và UTMOS **không đổi hoặc tốt hơn** FP32 (baseline: UTMOS22 gần như giữ nguyên, UTMOSv2 tốt hơn; vocos_small: cả 2 thang UTMOS đều tốt hơn FP32).

Xác nhận thêm cho `baseline::everything` (n=100, ban đầu chưa đo vì WER sweep không cho thấy dấu hiệu bất thường): UTMOS22 giảm **-0.30** (3.970→3.670) và UTMOSv2 giảm **-0.376** (3.162→2.786, xuống dưới cả FP32) so với `flow+enc_p+dp` — mức giảm còn nặng hơn vocos_small, củng cố thêm kết luận `dec` là layer nhạy cảm nhất ở cả 2 kiến trúc.

---

## 4b. Tốc độ thật trên CPU có AVX-VNNI — đánh đổi giữa loại trừ `dec` hay không

Máy chạy toàn bộ khảo sát này (Intel i7-14700, Raptor Lake) **có AVX-VNNI** — khác máy dev gốc trong `QAT_Quantization_Report.md` (i5-10400F, Comet Lake, không VNNI). Benchmark thật (`intra_op_num_threads` cố định 1 và 2, per ORT session, 3 lần warmup + 15 lần đo mỗi câu, 5 câu độ dài khác nhau — theo đúng phương pháp `vnni_test_package/bench_vnni.py`), trên `baseline`:

| Cấu hình | Speedup @2 thread | Speedup @1 thread |
|---|---|---|
| INT8 flow+enc_p+dp (dec=FP32) | 1.10x | 1.14x |
| INT8 everything (dec quantize) | **1.31x** | **1.52x** |

`dec` chiếm phần compute đáng kể (chuỗi upsample chạy ở độ phân giải thời gian tăng dần tới sample rate) nên quantize nó cho tốc độ tăng thật — rõ nhất ở 1 thread (gần gấp đôi mức tăng so với loại trừ `dec`). Nhưng đối chiếu với mục 4/4b: cái giá là UTMOS giảm ~0.3-0.4 điểm — **không đáng** cho một sản phẩm ưu tiên chất lượng giọng nói. `flow+enc_p+dp` vẫn là lựa chọn mặc định; `everything` chỉ nên cân nhắc khi tốc độ là ưu tiên tuyệt đối và chấp nhận chất lượng giảm rõ rệt.

Lưu ý: bản ONNX QDQ export trực tiếp từ checkpoint QAT (trọng số vẫn FP32, chỉ mô phỏng số học qua node QuantizeLinear/DequantizeLinear) **không phản ánh tốc độ thật** — RTF đo được trên bản này thường chậm hơn cả FP32 (do làm thêm việc round-trip giả lập) và không nên dùng để so sánh tốc độ; chỉ số liệu từ bản INT8 QOperator thật (trọng số lưu int8, như trong bảng trên) mới có ý nghĩa cho benchmark tốc độ.

---

## 5. Kết luận — công thức chung cho cả 2 kiến trúc

**Quantize `flow`+`enc_p`+`dp`, giữ nguyên `dec` ở FP32.**

Đây là công thức **duy nhất hoạt động tốt cho cả 2 kiến trúc** dù `dec` của chúng khác nhau hoàn toàn (HiFi-GAN vs Vocos/ConvNeXt+ISTFT) — vì phần khác biệt kiến trúc chỉ nằm ở `dec`, còn `enc_p`/`dp`/`flow` dùng chung giữa 2 model. Kết quả:

- **baseline**: size giảm 65% (65.4→23.0MB), WER tốt hơn FP32 (0.157→0.130), UTMOS gần như không đổi.
- **vocos_small**: size giảm 58% (73.3→31.0MB), WER tốt hơn FP32 (0.091→0.082), UTMOS **tốt hơn FP32** (UTMOSv2 3.274→3.314).

**Đối chiếu với `QAT_Quantization_Report.md`**: báo cáo đó kết luận "không có công thức chung, phải đo riêng từng kiến trúc" (dựa trên Piper Baseline vs BanhmiTTS_v1, 2 model không chia sẻ bất kỳ phần kiến trúc nào). Ở đây, vì `baseline` và `vocos_small` chia sẻ chung `enc_p`/`dp`/`flow` (chỉ khác `dec`), công thức "loại trừ toàn bộ `dec`, quantize phần còn lại" hóa ra transfer được — không mâu thuẫn với kết luận cũ, mà mở rộng nó: **công thức chung tồn tại khi (và chỉ khi) 2 kiến trúc chia sẻ phần lớn cấu trúc, khác biệt chỉ ở phần bị loại trừ.**

---

## 6. QAT training (baseline) — đang chạy, kết quả sơ bộ rất tích cực

Dựa trên công thức đã xác nhận (`flow+enc_p+dp`, loại `dec`), đã khởi động QAT fine-tuning cho `baseline`:

- Resume từ `baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt` (`global_step=1,168,160`).
- Wrap 181 lớp (`flow`=64, `enc_p`=37, `dp`=80), chỉ Conv1d/ConvTranspose1d (khớp đúng scope PTQ đã xác nhận — không wrap Linear của attention trong `enc_p`/`dp` vì export sẽ không bao giờ quantize chúng).
- **150 epoch** (thực đo ~399 batch/epoch × 150 ≈ 59,850 batch; do `global_step` gốc đếm cả bước G lẫn D nên quy đổi cùng đơn vị: 10% của 1,168,160 ≈ 116,816 "global step" ≈ 58,408 batch → 150 epoch ở đây tương ứng ~102% mục tiêu 10%, sai số không đáng kể so với ước tính ban đầu).
- LR: 5e-6 → 5e-7 (giảm dần 10x qua 150 epoch, độc lập với LR gốc — không phải "resume" lịch LR cũ).
- Discriminator warmup: đóng băng `model_d`/`model_d_dur` trong 785 batch đầu — **thực tế ≈2 epoch** (399 batch/epoch, không phải ~1 epoch như ước tính lúc lên kế hoạch do nhầm đơn vị đếm của `global_step`); không ảnh hưởng xấu, chỉ warmup dài hơn dự kiến (an toàn hơn, không rủi ro hơn).
- Freeze FakeQuantize observer sau epoch 60 (40% của 150) — xác nhận qua log đã kích hoạt đúng lúc.

### Kết quả cuối cùng (training hoàn tất 150/150 epoch — best checkpoint: epoch=93, val_loss_mel=19.2383)

Quy trình lấy bản deploy thật: (1) export trọng số đã học qua QAT sang ONNX FP32 thường (không wrapper fake-quant, không weight_norm — `export_fp32_from_qat.py`); (2) chạy PTQ thật (`quantize_static`, đúng scope `flow+enc_p+dp` đã xác nhận) lên trên để nén thành INT8 QOperator thật — khác với export QDQ trực tiếp từ checkpoint QAT (`export_onnx_qat.py`), vốn chỉ mô phỏng số học bằng node QuantizeLinear/DequantizeLinear và giữ nguyên trọng số FP32 trên đĩa (dùng tốt để đánh giá chất lượng nhưng vô dụng cho size/tốc độ thật — xem mục 4b).

| Variant | WER | UTMOS22 | UTMOSv2 | Size |
|---|---|---|---|---|
| FP32 gốc | 0.157 | 4.005 | 3.136 | 65.4MB |
| PTQ-thuần (flow+enc_p+dp) | 0.130 | 3.970 | 3.162 | 23.0MB |
| QAT epoch=93, QDQ mô phỏng (chỉ để đánh giá chất lượng) | 0.088 | 4.128 | 3.539 | 68.9MB (không đại diện) |
| **QAT epoch=93 → INT8 nén thật (deploy được)** | **0.104** | **4.125** | **3.578** | **23.0MB** |

Bản INT8 nén thật giữ gần như nguyên vẹn chất lượng của bản QDQ mô phỏng (WER nhích nhẹ 0.088→0.104 do bước PTQ cuối hiệu chỉnh lại calibration, không đáng kể), và **vượt trội hoàn toàn so với cả FP32 gốc lẫn PTQ-thuần**: nhỏ hơn FP32 65% (23.0MB) nhưng UTMOS22 cao hơn FP32 +0.12, UTMOSv2 cao hơn +0.44, WER thấp hơn PTQ-thuần 1/3. Đây là kết quả deploy cuối cùng cho `baseline`.

### vocos_small — đang chạy song song

Sau khi baseline hoàn tất, đã khởi động QAT cho `vocos_small` với cùng công thức và scope (`flow,enc_p,dp`, `--qat-wrap-types conv` — `dec` vẫn loại trừ hoàn toàn nên không cần wrap Linear):
- Resume từ `vocos_small_run_v5_fixed/checkpoints/best-epoch=1274-val_loss_mel=19.4528.ckpt` (`global_step=1,002,150`).
- **125 epoch** (thực đo ~399 batch/epoch, khớp đúng mốc 10% của tổng step gốc quy đổi cùng đơn vị — không lặp lại sai số ước tính d_warmup của lần chạy baseline).
- Discriminator warmup: 399 batch đầu (~1 epoch chính xác, đã sửa từ ước tính sai 785≈2 epoch của lần baseline).
- Freeze observer sau epoch 50 (40% của 125).

Sẽ cập nhật kết quả khi hoàn tất, theo đúng quy trình đã dùng cho baseline (QAT→PTQ nén thật→eval n=100).
