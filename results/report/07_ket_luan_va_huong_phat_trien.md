# 7. Kết luận và hướng phát triển

## 7.1. Kết luận lý thuyết

SARSA, Q-Learning và Expected SARSA chia sẻ cùng khung one-step TD control nhưng khác nhau ở bootstrap target. SARSA lấy mẫu action tiếp theo từ behavior policy. Q-Learning dùng giá trị cực đại và hướng tới greedy target policy. Expected SARSA tính kỳ vọng theo phân phối action của target policy.

Khác biệt nhỏ trong công thức dẫn tới các hệ quả quan trọng. SARSA phản ánh rủi ro do exploration trong target, nhưng chịu variance từ action sampling. Q-Learning tách target khỏi behavior policy và có bảo đảm hội tụ tabular về $Q^*$ dưới các điều kiện chuẩn, nhưng có thể gặp policy mismatch trong lúc học và maximization bias khi Q estimates nhiễu. Expected SARSA loại bỏ conditional variance do action sampling và có thể hoạt động on-policy hoặc off-policy, nhưng vẫn chịu randomness của reward và transition.

Không có thuật toán vượt trội tuyệt đối chỉ từ lý thuyết. Lựa chọn phụ thuộc mục tiêu: tối ưu greedy policy, giảm chi phí lỗi trong quá trình học, giảm biến động cập nhật, hoặc đạt hiệu năng nhanh dưới interaction budget hữu hạn.

## 7.2. Kết luận thực nghiệm

TODO [THỰC NGHIỆM]: Viết đoạn tổng kết Cliff Walking, nêu online return, cliff rate và greedy path.

TODO [THỰC NGHIỆM]: Viết đoạn tổng kết Frozen Lake, nêu TD-target variance và evaluation success rate trong hai cấu hình slippery.

TODO [THỰC NGHIỆM]: Viết đoạn tổng kết Taxi, nêu steps to threshold, final return, uncertainty và runtime.

Kết luận cuối phải ghi rõ giả thuyết nào được ủng hộ, giả thuyết nào bị bác bỏ và mức uncertainty. Không dùng các cụm "tốt nhất" hoặc "hội tụ nhanh nhất" nếu chỉ đúng cho một môi trường hoặc một metric.

## 7.3. Khuyến nghị lựa chọn thuật toán

| Điều kiện | Lựa chọn nên cân nhắc | Lý do |
|---|---|---|
| Lỗi exploration có chi phí cao | SARSA | Target phản ánh behavior policy và rủi ro action tiếp theo |
| Cần học greedy target từ dữ liệu exploratory | Q-Learning | Off-policy max backup |
| Action space nhỏ và muốn giảm action-sampling variance | Expected SARSA | Tính expectation thay cho một action sample |
| Q estimates nhiễu và overestimation đáng kể | Expected SARSA hoặc Double Q-Learning | Tránh hoặc giảm phụ thuộc vào một max estimator |
| Môi trường non-stationary | Cần đánh giá thêm với step-size cố định | Điều kiện hội tụ tiệm cận không còn là mục tiêu duy nhất |

Bảng trên là hướng dẫn có điều kiện, không thay thế kiểm thử trên bài toán cụ thể.

## 7.4. Hạn chế

Nghiên cứu giới hạn ở Q-table và ba môi trường rời rạc. Hyperparameter space chỉ được khảo sát một phần. Một policy hoặc threshold được xem là tốt trong benchmark chưa chắc phản ánh yêu cầu của ứng dụng thực tế. Ngoài ra, variance của TD target và variance của final performance không hoàn toàn tương đương.

Nếu chưa tính nghiệm tham chiếu, báo cáo không thể định lượng chính xác maximization bias. Nếu chỉ dùng epsilon decay với giới hạn dưới 0.01, kết quả cũng không phải minh chứng thực nghiệm trực tiếp cho GLIE hoặc hội tụ tiệm cận.

## 7.5. Hướng phát triển

### Double Q-Learning

Double Q-Learning tách action selection và action evaluation giữa hai bảng Q nhằm giảm maximization bias. Đây là mở rộng trực tiếp nhất nếu thí nghiệm phát hiện Q-Learning overestimate action value.

### N-step và eligibility traces

Các phương pháp n-step tạo cầu nối giữa one-step TD và Monte Carlo. SARSA($\lambda$) sử dụng eligibility traces để truyền TD error về nhiều trạng thái-hành động trước đó, có thể cải thiện credit assignment.

### Function approximation

Khi state space lớn hoặc liên tục, Q-table không còn phù hợp. Semi-gradient SARSA, linear function approximation và Deep Q-Network là các hướng mở rộng. Tuy nhiên, off-policy learning kết hợp bootstrapping và function approximation có thể gây bất ổn.

### Deadly Triad

Deadly Triad gồm function approximation, bootstrapping và off-policy training. Q-Learning đã có hai thành phần bootstrapping và off-policy; khi thêm function approximation, các bảo đảm tabular không còn áp dụng trực tiếp. Đây là hướng thảo luận quan trọng để nối kết báo cáo với deep reinforcement learning.

### Thiết kế benchmark rộng hơn

Có thể bổ sung Windy Gridworld, stochastic gridworld được kiểm soát hoặc maximization-bias MDP chuyên biệt. Một MDP nhỏ có nghiệm giải tích sẽ phù hợp hơn Taxi để đo trực tiếp overestimation.

## 7.6. Phát biểu kết thúc dự kiến

Kết quả của nghiên cứu cần được diễn giải từ bootstrap target tới hành vi quan sát được. SARSA, Q-Learning và Expected SARSA không chỉ là ba công thức cập nhật gần giống nhau; chúng biểu diễn ba lựa chọn khác nhau về policy được đánh giá và cách xử lý bất định ở bước kế tiếp. Việc lựa chọn thuật toán vì thế phải dựa trên loại rủi ro, nguồn stochasticity và mục tiêu đánh giá của bài toán, thay vì dựa trên một bảng xếp hạng duy nhất.
