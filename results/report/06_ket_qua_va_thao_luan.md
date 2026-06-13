# 6. Kết quả và thảo luận

> Chương này mới hoàn thiện phần cấu trúc và logic phân tích. Không điền nhận định định lượng trước khi chạy toàn bộ thí nghiệm.

## 6.1. Kiểm tra tính hợp lệ của thực nghiệm

Trước khi so sánh thuật toán, cần báo cáo:

- Phiên bản thư viện và cấu hình môi trường.
- Số seed hoàn thành và số run lỗi.
- Tổng environment steps thực tế.
- Tỷ lệ episode bị truncation.
- Kiểm tra evaluation không cập nhật Q.
- Kiểm tra cùng seed set và hyperparameter giữa các thuật toán.

TODO [THỰC NGHIỆM]: Điền bảng cấu hình cuối cùng và kết quả sanity check.

## 6.2. Cliff Walking: an toàn on-policy và policy mismatch

### 6.2.1. Kết quả cần trình bày

1. Biểu đồ online training return của SARSA và Q-Learning theo environment steps.
2. Biểu đồ greedy evaluation return.
3. Bảng cliff-fall rate, AUC, final return và confidence interval.
4. Hình greedy policy path sau huấn luyện.
5. Kết quả sensitivity của epsilon cố định và epsilon decay.

TODO [THỰC NGHIỆM]: Chèn hình và bảng.

### 6.2.2. Khung diễn giải

Nếu SARSA có online return tốt hơn và cliff rate thấp hơn nhưng Q-Learning có greedy evaluation path ngắn hơn, kết quả ủng hộ cơ chế policy mismatch: target của Q-Learning đánh giá greedy continuation trong khi behavior policy vẫn khám phá. Không nên diễn giải đây là Q-Learning "không học được"; hai metric đang đo hai policy khác nhau.

Nếu hai thuật toán không tạo đường đi khác biệt, cần kiểm tra learning rate, tie-breaking, episode budget và mức exploration. Cũng có thể $\epsilon$ hoặc penalty chưa đủ để làm rủi ro biểu hiện rõ. Nếu Q-Learning tốt hơn cả online return, giả thuyết H1 bị bác bỏ trong cấu hình đã dùng và báo cáo cần trình bày đúng kết quả đó.

TODO [THỰC NGHIỆM]: Viết kết luận Cliff Walking dựa trên effect size và CI.

## 6.3. Frozen Lake: action-sampling variance

### 6.3.1. Kết quả cần trình bày

1. TD-target variance theo training phase cho SARSA và Expected SARSA.
2. Evaluation success rate trên `is_slippery=False`.
3. Evaluation success rate trên `is_slippery=True`.
4. Final uncertainty across seeds.
5. Heatmap Q-value và, nếu có, sai số so với nghiệm Dynamic Programming.

TODO [THỰC NGHIỆM]: Chèn hình và bảng.

### 6.3.2. Khung diễn giải

Nếu Expected SARSA giảm TD-target variance trong cả hai cấu hình, kết quả phù hợp với việc tích phân qua action thay cho sampling. Nếu khoảng cách nhỏ hơn trong slippery setting hoặc variance tuyệt đối vẫn lớn, cần nhấn mạnh transition stochasticity chưa được loại bỏ.

Nếu TD-target variance thấp hơn nhưng success rate không cao hơn, điều đó không mâu thuẫn. Target ít biến động không bảo đảm policy tốt hơn trong mọi budget; sparse reward, exploration và bias hữu hạn mẫu vẫn ảnh hưởng. Ngược lại, nếu reward curve có vẻ mượt hơn nhưng M8 không giảm, không nên dùng cảm nhận thị giác để xác nhận giả thuyết.

TODO [THỰC NGHIỆM]: Viết kết luận Frozen Lake và phân biệt local update variance với across-seed performance variance.

## 6.4. Taxi: sample efficiency và độ ổn định

### 6.4.1. Kết quả cần trình bày

1. Distribution của steps to first success.
2. Steps to evaluation threshold và tỷ lệ seed đạt threshold.
3. Evaluation learning curve và normalized AUC.
4. Final return, success rate và illegal-action rate.
5. Wall-clock update time.

TODO [THỰC NGHIỆM]: Chèn hình và bảng.

### 6.4.2. Khung diễn giải

Nếu Q-Learning đạt threshold sớm hơn nhưng Expected SARSA có CI hẹp hơn ở cuối training, kết quả ủng hộ sự đánh đổi giữa greedy target và expectation target trong cấu hình này. Tuy nhiên, không được suy rộng thành định luật chung cho mọi sparse-reward task.

Nếu Expected SARSA học nhanh hơn, có thể expectation target tận dụng tốt hơn toàn bộ phân phối action hoặc Q-Learning chịu nhiễu từ toán tử max. Muốn gán nguyên nhân cho maximization bias cần phép đo overestimation so với giá trị tham chiếu; chỉ reward curve là chưa đủ.

TODO [THỰC NGHIỆM]: Viết kết luận Taxi và nêu rõ giả thuyết H3 được ủng hộ hay bác bỏ.

## 6.5. Sensitivity analysis

### Learning rate

TODO [THỰC NGHIỆM]: Báo AUC, final return và update variance cho từng $\alpha$.

Phân tích cần phân biệt ba hiện tượng: học chậm do $\alpha$ nhỏ, dao động do $\alpha$ lớn và divergence thực sự. Learning rate cố định không thỏa Robbins-Monro cho hội tụ tiệm cận, nhưng nghiên cứu vẫn có thể đánh giá hiệu năng trong ngân sách hữu hạn.

### Exploration schedule

TODO [THỰC NGHIỆM]: So sánh fixed epsilon và decay bằng online return, cliff rate và greedy evaluation.

Nếu gap SARSA-Q-Learning thu hẹp khi epsilon giảm, kết quả phù hợp với việc behavior policy tiến gần greedy target. Không gọi lịch có `epsilon_min=0.01` là GLIE.

## 6.6. Tổng hợp giữa các môi trường

TODO [THỰC NGHIỆM]: Hoàn thiện bảng sau.

| Trục nghiên cứu | Môi trường | So sánh | Kết quả | Mức ủng hộ giả thuyết |
|---|---|---|---|---|
| Behavior-target safety | Cliff Walking | SARSA vs Q-Learning | TODO | TODO |
| Action-sampling variance | Frozen Lake | SARSA vs Expected SARSA | TODO | TODO |
| Sample efficiency/stability | Taxi | Q-Learning vs Expected SARSA | TODO | TODO |

## 6.7. Đe dọa tới tính hợp lệ

### Internal validity

Kết quả có thể bị ảnh hưởng bởi bug trong terminal handling, khác seed, khác exploration schedule hoặc evaluation làm thay đổi Q-table. Các sanity check và unit test phải được thực hiện trước khi phân tích.

### Construct validity

Moving average reward không đồng nhất với hội tụ toán học. First success không đại diện đầy đủ cho sample efficiency. Heatmap trực quan không phải thước đo accuracy. Vì vậy, mỗi khái niệm được đo bằng nhiều chỉ số có vai trò khác nhau.

### External validity

Ba môi trường đều là tabular benchmark nhỏ. Kết luận không tự động mở rộng sang continuous state space, function approximation hoặc deep RL. Đặc biệt, tính ổn định trong tabular setting không phản ánh đầy đủ Deadly Triad.

### Statistical conclusion validity

Số seed hữu hạn làm confidence interval còn rộng. Metrics first-success có thể lệch mạnh. Nếu lựa chọn threshold hoặc smoothing sau khi xem dữ liệu, kết luận có nguy cơ thiên lệch. Báo cáo cần lưu cấu hình phân tích trước và công bố mọi run.
