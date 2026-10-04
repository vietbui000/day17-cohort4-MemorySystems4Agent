# Phân tích kết quả Lab 17

Chạy `python src/benchmark.py` ở chế độ offline, với hai agent dùng cùng dữ liệu trong `data/`. Số token là ước lượng theo độ dài ký tự, không phải số token được nhà cung cấp LLM tính phí.

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1274 | 13598 | 0.0% | 0.0% | 0 | 0 |
| Standard | Advanced | 2512 | 24254 | 64.3% | 63.8% | 425 | 0 |
| Long-context stress | Baseline | 485 | 22057 | 0.0% | 0.0% | 0 | 0 |
| Long-context stress | Advanced | 894 | 18535 | 50.0% | 50.0% | 731 | 1 |

## 1. Vì sao Advanced nhớ xuyên phiên tốt hơn?

Baseline chỉ giữ message theo `thread_id`, nên các câu hỏi recall được hỏi ở thread mới cho kết quả 0%. Advanced trích fact từ lời người dùng rồi cập nhật `state/profiles/<user>/User.md`. Khi sang thread mới, nó đọc lại hồ sơ đó. Vì vậy recall đạt 64,3% trên Standard và 50% trên stress. Các mức này vẫn thấp: regex chưa nhận ra mọi cách nói tự nhiên, và câu trả lời offline chỉ liệt kê fact đã trích được. Việc thay fact theo khóa xử lý một số correction, nhưng chưa phải nhận diện xung đột đầy đủ.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Standard không có lần compact nào. Advanced thêm nội dung `User.md` vào ngữ cảnh ở mỗi lượt; prompt tokens processed là 24.254, so với 13.598 của Baseline. Câu trả lời offline của Advanced cũng thường liệt kê nhiều fact, nên agent tokens only là 2.512 so với 1.274. Persistent memory cải thiện recall nhưng tạo chi phí đọc hồ sơ và trả lời dài hơn; không có lợi thế nén trong bộ dữ liệu này.

## 3. Compact giúp gì ở hội thoại dài?

Trong stress benchmark, `CompactMemoryManager` nén các message cũ thành summary và giữ 6 message gần nhất khi vượt ngưỡng 2.000 token. Một lần compact làm tổng prompt tokens processed của Advanced còn 18.535, thấp hơn Baseline 3.522 token ước lượng (khoảng 16%). Đây là lợi thế ở **ngữ cảnh phải xử lý lặp lại**, không đồng nghĩa giảm mọi loại usage: agent tokens only của Advanced vẫn cao hơn (894 so với 485). Summary giới hạn độ dài nên có thể bỏ mất chi tiết cần cho câu hỏi sau; 50% recall ở stress cho thấy việc tiết kiệm ngữ cảnh chưa bảo đảm nhớ đủ.

## 4. Memory file tăng thế nào và có rủi ro gì?

`User.md` tăng thêm 425 byte ở Standard và 731 byte ở stress; Baseline không tạo file. `upsert_fact()` thay giá trị cùng khóa, nên một correction được nhận diện có thể thay fact cũ mà không tăng thêm dòng. Tuy nhiên, bộ trích xuất theo regex có thể bỏ sót fact, ghi nhầm câu nói giả định thành fact, hoặc ghi một giá trị quá dài. Hồ sơ đó được đưa lại vào prompt các lượt sau, khiến lỗi tồn tại xuyên phiên và tăng chi phí. Cần kiểm tra fact nhạy với correction và cân nhắc ngưỡng tin cậy nếu phát triển tiếp.

## Giới hạn phép đo

`Cross-session recall` chấm mức khớp chuỗi kỳ vọng theo 0/0,5/1; `Response quality` là tỷ lệ chuỗi kỳ vọng xuất hiện. Hai chỉ số gần nhau và không đánh giá độ tự nhiên, tính liên quan, hay fact sai xuất hiện cùng câu trả lời. Benchmark chạy offline, không đo token thực, latency, chi phí API hoặc chất lượng LLM live. Các kết luận trên chỉ áp dụng cho hai tập dữ liệu cố định trong repo.
