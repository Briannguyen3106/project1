# 5. Phương pháp đánh giá

## 5.1. Nguyên tắc chung

Đánh giá phải trả lời riêng ba câu hỏi: tác tử học như thế nào trong quá trình tương tác, greedy policy hiện tại tốt đến đâu, và kết quả có ổn định giữa các run hay không. Vì vậy, báo cáo không dùng một đường training reward duy nhất để đại diện cho toàn bộ chất lượng thuật toán.

Trục ngang ưu tiên là cumulative environment steps. Episode vẫn được báo cáo để dễ diễn giải, nhưng không dùng làm đơn vị sample efficiency duy nhất vì độ dài episode phụ thuộc thuật toán.

## 5.2. Online training return

Return của episode huấn luyện là:

$$
G^{\text{train}}_i=\sum_{t=0}^{T_i-1}\gamma^tR_{t+1}.
$$

Nếu môi trường benchmark thường báo undiscounted episodic return, cần lưu thêm tổng reward không chiết khấu để so sánh với quy ước chuẩn. Online return phản ánh hiệu năng của behavior policy, bao gồm exploration và các lỗi xảy ra trong lúc học.

Moving average cửa sổ $w$ được tính:

$$
\bar{G}_i=\frac{1}{w}\sum_{j=i-w+1}^{i}G^{\text{train}}_j.
$$

Cửa sổ dự kiến là 50 episode, nhưng hình theo environment steps nên sử dụng bin hoặc smoothing có quy tắc giống nhau giữa thuật toán. Đường chưa làm mượt hoặc raw summary cần được lưu để tránh smoothing che mất biến động.

## 5.3. Greedy evaluation

Tại các checkpoint cố định, policy được đánh giá bằng nhiều episode với $\epsilon=0$ và không cập nhật Q. Evaluation mean tại checkpoint $k$ là:

$$
\hat{J}_k=\frac{1}{N_{eval}}\sum_{i=1}^{N_{eval}}G_{k,i}^{eval}.
$$

Đối với Frozen Lake, success rate thường có ý nghĩa trực tiếp hơn reward mean vì reward là nhị phân. Đối với Cliff Walking, evaluation cho biết chất lượng greedy path trong khi training return phản ánh rủi ro dưới behavior policy.

## 5.4. Steps to first success

Metric này là cumulative environment steps trước khi xuất hiện episode thành công đầu tiên. Nó phù hợp với Frozen Lake và Taxi, nhưng có phân phối lệch và dễ bị outlier. Vì vậy cần báo median, interquartile range hoặc empirical distribution bên cạnh mean. Run không thành công trong budget phải được xử lý như censored observation hoặc gán rõ giá trị vượt budget; không được loại khỏi thống kê.

## 5.5. Steps to threshold

Một thuật toán đạt threshold tại checkpoint đầu tiên mà evaluation metric vượt $\theta$ và duy trì yêu cầu ổn định đã định trước. Threshold dự kiến:

- Cliff Walking: evaluation return $\theta=-20$.
- Frozen Lake: success rate $\theta=0.7$.
- Taxi: evaluation return $\theta=5$.

Các ngưỡng phải được xác định trước khi xem kết quả chính và kiểm tra bằng random baseline cùng một policy gần tối ưu. Báo cáo cần nêu tỷ lệ seed đạt threshold. Chỉ báo mean time-to-threshold trên các seed thành công sẽ tạo survivorship bias.

## 5.6. Learning-curve AUC

AUC đo hiệu năng tích lũy trong toàn bộ quá trình học. Với các checkpoint $(x_k,J_k)$ theo environment steps, có thể dùng quy tắc hình thang:

$$
\operatorname{AUC}
=\sum_{k=1}^{K-1}
\frac{J_k+J_{k+1}}{2}(x_{k+1}-x_k).
$$

Để dễ so sánh giữa experiment budget khác nhau, AUC có thể chia cho tổng step range. AUC bổ sung cho final performance: hai thuật toán có cùng kết quả cuối nhưng thuật toán học sớm hơn sẽ có AUC cao hơn.

## 5.7. Uncertainty across seeds

Với metric cuối $X_1,\ldots,X_n$ từ $n$ seed, mean và sample standard deviation là:

$$
\bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_i,
$$

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar{X})^2}.
$$

Khoảng tin cậy 95% có thể được ước lượng bằng bootstrap percentile trên seed-level metric. Bootstrap phù hợp hơn normal approximation khi số seed không lớn hoặc phân phối lệch. Resampling phải diễn ra ở cấp seed, không resample từng episode như thể chúng độc lập.

Biểu đồ learning curve nên vẽ mean và dải 95% confidence interval across seeds tại từng checkpoint. Cần ghi rõ dải là CI, standard error hay standard deviation.

## 5.8. TD-target variance

Để kiểm chứng trực tiếp giả thuyết SARSA và Expected SARSA, lưu target $Y_t$ và TD error $\delta_t$. Trong mỗi cửa sổ step, tính sample variance:

$$
\widehat{\operatorname{Var}}(Y)
=\frac{1}{m-1}\sum_{j=1}^{m}(Y_j-\bar{Y})^2.
$$

Variance gộp trên mọi trạng thái có thể bị ảnh hưởng bởi việc hai thuật toán thăm phân phối trạng thái khác nhau. Do đó nên bổ sung một trong hai cách:

- Báo variance có điều kiện trên một nhóm trạng thái được thăm đủ nhiều.
- Chạy một evaluation dataset cố định gồm transition/state và tính target của hai Q-table trên cùng dữ liệu.

Mục tiêu là tách ảnh hưởng của target estimator khỏi thay đổi state visitation càng nhiều càng tốt.

## 5.9. Safety và error rate

Trong Cliff Walking:

$$
\text{Cliff rate}
=\frac{\text{số lần rơi vách}}{\text{tổng environment steps}}
$$

hoặc báo số lần rơi trên 100 episode. Trong Taxi, illegal pickup/dropoff rate được định nghĩa tương tự. Denominator phải nhất quán giữa thuật toán.

Safety metric cần được trình bày cho training behavior. Greedy evaluation có thể không rơi vách dù behavior policy chịu rủi ro lớn; gộp hai chế độ sẽ che khuất hiện tượng cần nghiên cứu.

## 5.10. Policy path và Q-value heatmap

Policy path trên Cliff Walking được tạo bằng greedy rollout từ start. Nếu policy lặp vô hạn, visualization phải dừng ở step cap và đánh dấu loop. Nếu có tie, dùng cùng random tie-breaking hoặc một quy tắc cố định đã công bố.

Heatmap Frozen Lake hiển thị:

$$
V_Q(s)=\max_aQ(s,a).
$$

Heatmap giúp quan sát cấu trúc value estimate nhưng không đủ để kết luận accuracy. Nếu có mô hình transition, dùng value iteration để tính $V^*$ và báo RMSE hoặc Bellman residual:

$$
\operatorname{RMSE}(V_Q,V^*)
=\sqrt{\frac{1}{|\mathcal{S}|}\sum_s(V_Q(s)-V^*(s))^2}.
$$

## 5.11. Chi phí tính toán

Sample efficiency được đo bằng environment steps; computational efficiency được đo bằng thời gian. Hai khái niệm không được trộn lẫn. Có thể báo:

- Tổng wall-clock training time.
- Thời gian trung bình cho một update.
- Số update mỗi giây khi loại trừ hoặc kiểm soát thời gian environment.

Q-Learning và Expected SARSA đều thường duyệt $|A|$ giá trị. Expected SARSA thực hiện thêm phép nhân với xác suất policy; chênh lệch thực tế cần được đo thay vì suy ra chỉ từ Big-O khi $|A|=6$.

## 5.12. Trình bày thống kê

Mỗi bảng kết quả cuối cần có:

| Môi trường | Thuật toán | Số seed | Metric | Mean | Std | 95% CI |
|---|---|---:|---|---:|---:|---:|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Nếu thực hiện kiểm định giả thuyết, nên ưu tiên effect size và confidence interval hơn việc chỉ báo p-value. Nhiều metric và nhiều cặp so sánh làm tăng nguy cơ multiple-comparison; báo cáo cần xác định trước metric chính của từng môi trường.

## 5.13. Metric chính theo môi trường

| Môi trường | Metric chính | Metric hỗ trợ |
|---|---|---|
| Cliff Walking | Online return, cliff-fall rate | Greedy return, path length, AUC |
| Frozen Lake | TD-target variance, evaluation success rate | First success, Bellman error, heatmap |
| Taxi | Steps to threshold, final evaluation return | First success, illegal-action rate, AUC, runtime |
