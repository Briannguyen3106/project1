# Bộ nội dung báo cáo Temporal Difference Control

Thư mục này chứa các chương độc lập của báo cáo **Nghiên cứu và đánh giá sự khác biệt cốt lõi giữa SARSA, Q-Learning và Expected SARSA**. Nội dung được xây dựng theo khung trong `content.md` và phần lý thuyết Temporal-Difference Learning của tài liệu `Temporal_Difference_Learning.pdf`.

## Thứ tự ghép báo cáo

1. `01_mo_dau.md`
2. `02_co_so_ly_thuyet.md`
3. `03_ba_thuat_toan_td_control.md`
4. `04_thiet_ke_thuc_nghiem.md`
5. `05_phuong_phap_danh_gia.md`
6. `06_ket_qua_va_thao_luan.md`
7. `07_ket_luan_va_huong_phat_trien.md`

## Quy ước biên tập

- Ký hiệu `TODO [THỰC NGHIỆM]` đánh dấu nội dung chỉ có thể hoàn thiện sau khi chạy code.
- Các nhận định ở chương thiết kế thực nghiệm là **giả thuyết nghiên cứu**, không phải kết luận định trước.
- Kết quả huấn luyện của behavior policy và kết quả đánh giá greedy policy phải được trình bày riêng.
- Mọi bảng số liệu cuối cùng cần ghi số seed, trung bình, độ lệch chuẩn hoặc khoảng tin cậy 95%.
- Không dùng mức độ "mượt" của đường cong hoặc heatmap như bằng chứng định lượng nếu chưa có metric hỗ trợ.

## Nguồn lý thuyết

Tài liệu PDF được cung cấp là bản scan không có lớp văn bản để trích dẫn tự động theo số trang. Trước khi xuất bản bản cuối, cần đối chiếu thủ công số trang của các nội dung sau: TD prediction, on-policy control (SARSA), off-policy control (Q-Learning), Expected SARSA và Cliff Walking. Phần tài liệu tham khảo cuối báo cáo nên bổ sung đầy đủ tác giả, năm xuất bản, tên sách/tài liệu và số chương hoặc số trang.
