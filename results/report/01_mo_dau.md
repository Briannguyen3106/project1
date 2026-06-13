# 1. Mở đầu

## 1.1. Bối cảnh nghiên cứu

Học tăng cường (Reinforcement Learning, RL) nghiên cứu cách một tác tử học hành vi thông qua tương tác tuần tự với môi trường. Ở mỗi thời điểm, tác tử quan sát trạng thái, chọn hành động, nhận phần thưởng và chuyển sang trạng thái mới. Khác với học có giám sát, tác tử không được cung cấp nhãn đúng cho từng quyết định. Thay vào đó, nó phải tìm một chiến lược hành động nhằm tối đa hóa tổng phần thưởng tích lũy trong dài hạn.

Một khó khăn trung tâm của RL là phần thưởng có thể bị trì hoãn. Một hành động tốt ở hiện tại không nhất thiết tạo ra phần thưởng tức thời, nhưng có thể đưa tác tử tới những trạng thái thuận lợi hơn trong tương lai. Vì vậy, thuật toán phải giải quyết bài toán gán công trạng theo thời gian: xác định mức độ đóng góp của mỗi quyết định vào kết quả dài hạn.

Trong trường hợp không gian trạng thái và hành động hữu hạn, giá trị của từng trạng thái hoặc cặp trạng thái-hành động có thể được lưu trực tiếp trong bảng. Cách biểu diễn này được gọi là học tăng cường dạng bảng (tabular reinforcement learning). Mặc dù không mở rộng tốt tới không gian rất lớn, tabular RL có vai trò quan trọng vì cho phép quan sát rõ cơ chế cập nhật, phân tích điều kiện hội tụ và kiểm soát các nguồn sai số mà không bị nhiễu bởi xấp xỉ hàm.

Ba họ phương pháp kinh điển để ước lượng hàm giá trị là Dynamic Programming (DP), Monte Carlo (MC) và Temporal-Difference (TD). DP sử dụng mô hình chuyển trạng thái và phần thưởng của môi trường để thực hiện phép kỳ vọng đầy đủ. MC không cần mô hình nhưng phải chờ episode kết thúc mới có thể tính return thực tế. TD nằm giữa hai cách tiếp cận: thuật toán học trực tiếp từ trải nghiệm như MC, đồng thời bootstrap từ ước lượng hiện tại giống DP. Nhờ vậy, TD có thể cập nhật sau từng bước và áp dụng cho cả bài toán liên tục hoặc episode dài.

Khi TD được mở rộng từ prediction sang control, mục tiêu không chỉ là đánh giá một policy cố định mà còn phải cải thiện policy trong quá trình học. SARSA, Q-Learning và Expected SARSA là ba thuật toán TD control một bước có hình thức gần giống nhau. Điểm khác biệt chủ yếu nằm ở thành phần bootstrap của TD target. Tuy chỉ thay đổi một biểu thức, lựa chọn này quyết định quan hệ giữa behavior policy và target policy, nguồn phương sai trong cập nhật, khả năng xuất hiện maximization bias và hành vi rủi ro của tác tử trong lúc học.

## 1.2. Vấn đề nghiên cứu

Việc chỉ so sánh reward cuối cùng của ba thuật toán trên một benchmark tổng quát không đủ để giải thích bản chất khác biệt giữa chúng. Một thuật toán có thể tạo ra greedy policy tốt sau huấn luyện nhưng chịu nhiều lỗi trong quá trình khám phá. Một thuật toán khác có thể cập nhật ổn định hơn nhưng cần nhiều tương tác hơn để đạt cùng mức hiệu năng. Nếu training return, evaluation return, bias và variance bị gộp vào một chỉ số duy nhất, kết luận dễ trở nên thiếu chính xác.

Báo cáo này tiếp cận vấn đề bằng cách xem bootstrap target là biến số trung tâm. Ba target được khảo sát là:

$$
Y_t^{\text{SARSA}}=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),
$$

$$
Y_t^{\text{Q}}=R_{t+1}+\gamma\max_a Q(S_{t+1},a),
$$

và

$$
Y_t^{\text{Expected}}=R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a).
$$

Từ ba biểu thức này, báo cáo phân tích bốn câu hỏi chính:

1. Việc bootstrap theo action thực sự của behavior policy ảnh hưởng thế nào đến độ an toàn trong lúc học?
2. Thay action sample bằng kỳ vọng theo policy làm giảm thành phần phương sai nào của TD target?
3. Toán tử cực đại giúp Q-Learning hướng tới greedy target nhưng có thể tạo ra sai lệch lạc quan trong điều kiện nào?
4. Dưới cùng ngân sách tương tác và cùng cơ chế exploration, thuật toán nào đạt ngưỡng hiệu năng sớm hơn và thuật toán nào ổn định hơn giữa các seed?

## 1.3. Mục tiêu nghiên cứu

Mục tiêu tổng quát là làm rõ sự khác biệt về lý thuyết và hành vi thực nghiệm giữa SARSA, Q-Learning và Expected SARSA trong tabular setting.

Các mục tiêu cụ thể gồm:

- Hệ thống hóa nền tảng MDP, hàm giá trị, phương trình Bellman, TD target và TD error.
- Trình bày cơ chế cập nhật và quan hệ on-policy/off-policy của từng thuật toán.
- Phân tích riêng các thành phần bias, variance, policy mismatch và chi phí tính target.
- Xây dựng ba nhóm thí nghiệm có mục tiêu phân biệt rõ từng cặp thuật toán.
- Tách hiệu năng trong quá trình huấn luyện khỏi chất lượng greedy policy sau hoặc trong các lần đánh giá định kỳ.
- Đánh giá độ nhạy theo learning rate và exploration schedule.
- Đưa ra kết luận có điều kiện thay vì tìm một thuật toán vượt trội tuyệt đối.

## 1.4. Câu hỏi và giả thuyết nghiên cứu

**RQ1.** Với $\epsilon$ cố định trong Cliff Walking, SARSA có tạo ra behavior policy an toàn hơn Q-Learning không?

**H1.** SARSA được kỳ vọng có tỷ lệ rơi vách thấp hơn và online training return tốt hơn, vì target của nó phản ánh cả xác suất thực hiện hành động khám phá ở trạng thái kế tiếp. Q-Learning có thể học greedy path ngắn hơn nhưng behavior policy vẫn có khả năng rơi vách khi đi sát vùng nguy hiểm.

**RQ2.** Expected SARSA có làm giảm biến động cập nhật so với SARSA trong Frozen Lake không?

**H2.** Expected SARSA được kỳ vọng có conditional variance của TD target thấp hơn SARSA vì nó tích phân chính xác qua phân phối action. Tuy nhiên, khác biệt này không loại bỏ randomness từ reward hoặc transition.

**RQ3.** Q-Learning và Expected SARSA khác nhau thế nào về sample efficiency và độ ổn định trên Taxi?

**H3.** Greedy target của Q-Learning có thể giúp đạt thành công sớm hơn, trong khi target kỳ vọng của Expected SARSA có thể tạo kết quả ít biến động hơn giữa các seed. Đây là giả thuyết cần được dữ liệu kiểm chứng; không giả định trước thuật toán nào chắc chắn thắng.

## 1.5. Phạm vi nghiên cứu

Nghiên cứu giới hạn ở MDP hữu hạn, action space rời rạc và biểu diễn Q-table. Ba môi trường khảo sát là `CliffWalking-v0`, `FrozenLake-v1` và `Taxi-v3`. Báo cáo không triển khai function approximation, replay buffer, target network hoặc deep reinforcement learning. Các chủ đề này chỉ được đề cập ở phần hướng phát triển.

Phạm vi lý thuyết tập trung vào one-step TD control. Các biến thể nhiều bước, eligibility traces và n-step Expected SARSA không thuộc nội dung thực nghiệm chính. Maximization bias được thảo luận như một hệ quả có thể có của toán tử cực đại; chỉ được kết luận thực nghiệm nếu có phép đo sai lệch so với giá trị tham chiếu.

## 1.6. Đóng góp dự kiến

Đóng góp của báo cáo nằm ở thiết kế so sánh có kiểm soát. Thay vì dùng ba môi trường như các benchmark chung, mỗi môi trường được gắn với một câu hỏi nghiên cứu cụ thể. Báo cáo đồng thời chuẩn hóa cách đo theo environment steps, sử dụng nhiều seed, trình bày uncertainty và tách online behavior khỏi greedy evaluation. Cách tổ chức này giúp liên hệ trực tiếp kết quả quan sát được với thành phần khác nhau trong TD target.

## 1.7. Cấu trúc báo cáo

Chương 2 trình bày nền tảng MDP, Bellman equation và Temporal-Difference learning. Chương 3 phân tích SARSA, Q-Learning và Expected SARSA. Chương 4 mô tả môi trường, giả thuyết và quy trình thực nghiệm. Chương 5 định nghĩa các metric và phương pháp tổng hợp thống kê. Chương 6 dành cho kết quả và thảo luận sau khi chạy thí nghiệm. Chương 7 tổng kết, nêu giới hạn và hướng phát triển.
