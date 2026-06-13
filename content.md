
---

# NGHIÊN CỨU VÀ ĐÁNH GIÁ SỰ KHÁC BIỆT CỐT LÕI GIỮA CÁC THUẬT TOÁN TEMPORAL DIFFERENCE CONTROL: SARSA, Q-LEARNING VÀ EXPECTED SARSA

---

## 1. MỞ ĐẦU

**1.1. Bối cảnh nghiên cứu**
Giới thiệu về Học tăng cường (RL) và bài toán ước lượng hàm giá trị dạng bảng (Tabular RL). Vị trí của TD trong tam giác DP – MC – TD.

**1.2. Lý do chọn đề tài**
Tầm quan trọng của chiến lược cập nhật (On-policy vs. Off-policy) trong thực tế. Sự đánh đổi giữa tính an toàn, tốc độ hội tụ và ổn định phương sai — ba trục mà bộ ba thuật toán phân kỳ rõ nhất.

**1.3. Mục tiêu**
Phân tích khác biệt trong bootstrap target của ba thuật toán và đánh giá thực nghiệm có kiểm soát trên ba môi trường. Trọng tâm là kiểm chứng cách target ảnh hưởng đến policy mismatch, bias, conditional variance, độ an toàn và sample efficiency dưới cùng một ngân sách tương tác với môi trường.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ ĐẢM BẢO TOÁN HỌC

**2.1. MDP và Phương trình Bellman**
Tóm tắt $(S, A, P, R, \gamma)$. Phương trình Bellman optimality cho $Q^*$.

**2.2. Ý tưởng cốt lõi của Temporal Difference**
- Cơ chế bootstrapping — cập nhật cuốn chiếu không cần đợi kết thúc episode
- TD Target và TD Error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

**2.3. Đảm bảo toán học về hội tụ**
Điều kiện Robbins-Monro cho $\alpha_t$:

$$
\sum \alpha_t = \infty,\qquad \sum \alpha_t^2 < \infty
$$

Đây là điều kiện cần của stochastic approximation, không phải điều kiện đủ. Điều kiện đủ cho hội tụ còn cần thêm tính chất co của Bellman operator và mức độ exploration thích hợp.

Toán tử Bellman optimality là toán tử co (contraction mapping) trong không gian $\ell^\infty$ với hệ số $\gamma < 1$ → tồn tại điểm cố định duy nhất $Q^*$.

Điều kiện GLIE (Greedy in the Limit with Infinite Exploration) yêu cầu mọi cặp $(s,a)$ tiếp tục được khám phá vô hạn lần, đồng thời policy tiến dần về greedy. Đây là điều kiện quan trọng trong các định lý hội tụ tabular, nhưng phải được xét cùng Robbins-Monro và các giả định MDP hữu hạn. Một lịch $\epsilon$ giảm đến một giá trị dương cố định không thỏa GLIE.

---

## 3. PHÂN TÍCH CHI TIẾT BA THUẬT TOÁN TD CONTROL

**3.1. SARSA — On-policy TD Control**
- Công thức cập nhật: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]$
- Tuple $(S, A, R, S', A')$ — hành động $A_{t+1}$ được lấy mẫu từ policy hiện tại
- Với GLIE, Robbins-Monro và các giả định tabular phù hợp, SARSA hội tụ về $Q^*$. Với $\epsilon$ cố định, thuật toán tiếp tục đánh giá và cải thiện một policy $\epsilon$-soft; không nên đồng nhất trường hợp này với bảo đảm hội tụ GLIE về $Q^*$

**3.2. Q-Learning — Off-policy TD Control**
- Công thức cập nhật: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)\right]$
- Toán tử $\max_a$ tách biệt hoàn toàn target policy (greedy) khỏi behavior policy ($\epsilon$-greedy)
- Trong tabular setting, hội tụ về $Q^*$ nếu mọi cặp $(s,a)$ được thăm vô hạn lần, step-size thỏa Robbins-Monro và các giả định chuẩn được đáp ứng
- Toán tử $\max$ có thể gây maximization bias khi các ước lượng action value chứa nhiễu; đây là giả thuyết cần đo thay vì mặc định xuất hiện ở mọi môi trường stochastic

**3.3. Expected SARSA — Expected TD Control**
- Công thức cập nhật: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right]$
- Thay thế sample $Q(S_{t+1}, A_{t+1})$ bằng kỳ vọng giải tích → loại bỏ conditional variance do lấy mẫu $A_{t+1}$, nhưng không loại bỏ randomness từ reward và transition
- Tổng quát hóa: với $\pi$ greedy hoàn toàn → tương đương Q-Learning; với $\pi$ = behavior policy → on-policy
- Trong project này, Expected SARSA dùng expectation theo chính behavior policy $\epsilon$-greedy, trừ khi một thí nghiệm được ghi rõ là off-policy
- Chi phí tính target là $O(|A|)$; Q-Learning cũng thường cần $O(|A|)$ để tính $\max_a Q(S',a)$, còn SARSA chỉ cần tra cứu action đã lấy mẫu

**3.4. Bảng tổng hợp so sánh lý thuyết**

| Tiêu chí | SARSA | Q-Learning | Expected SARSA |
|---|---|---|---|
| TD Target | $R + \gamma Q(S', A')$ | $R + \gamma \max_a Q(S', a)$ | $R + \gamma \sum_a \pi(a|S') Q(S', a)$ |
| On/Off-policy | On-policy | Off-policy | Cả hai (tùy $\pi$) |
| Toán tử cập nhật | Sample | Max | Expectation |
| Variance do lấy mẫu $A'$ | Có | Không | Không |
| Maximization bias | Không do toán tử max | Có thể có | Không do toán tử max nếu expectation theo policy non-greedy |
| Hội tụ tabular | $Q^*$ dưới GLIE và các điều kiện chuẩn | $Q^*$ dưới exploration vô hạn và các điều kiện chuẩn | Phụ thuộc target policy; về $Q^*$ trong các cấu hình phù hợp |
| Chi phí tính target | $O(1)$ | $O(|A|)$ | $O(|A|)$ |

---

## 4. THIẾT KẾ MÔI TRƯỜNG THỰC NGHIỆM

> **Nguyên tắc thiết kế:** Mỗi môi trường được chọn để **phân biệt một cặp thuật toán cụ thể** trên một trục lý thuyết rõ ràng. Không có môi trường nào là "general benchmark".

**4.1. Môi trường 1 — Cliff Walking (`CliffWalking-v0`)**
*Trục kiểm tra: On-policy vs Off-policy safety*
*Cặp được phân biệt: SARSA vs Q-Learning*

- Không gian: lưới $4 \times 12$, vách đá phạt $-100$
- Cơ chế bẫy: với $\epsilon > 0$, Q-Learning học policy greedy đi sát vách (ngắn nhất) nhưng khi thực thi behavior policy $\epsilon$-greedy thì thỉnh thoảng rơi vách → reward trung bình thấp hơn SARSA
- SARSA học policy an toàn hơn vì target phụ thuộc hành động thực thi
- **Lưu ý thiết kế:** Giữ $\epsilon$ **cố định** (không decay) ở bước phân tích này để hiện tượng phân kỳ giữa hai thuật toán còn rõ. Epsilon-decay chỉ dùng trong sensitivity analysis (Section 5.3)

**4.2. Môi trường 2 — Frozen Lake (`FrozenLake-v1`, `is_slippery=True`)**
*Trục kiểm tra: Variance dưới môi trường stochastic*
*Cặp được phân biệt: Expected SARSA vs SARSA*

- Cơ chế trượt: với xác suất $1/3$, agent di chuyển theo hướng vuông góc thay vì hướng chọn → $P(s'|s,a)$ không deterministic
- **Giả thuyết:** Expected SARSA có conditional variance của TD target thấp hơn SARSA vì loại bỏ sampling $A_{t+1}$; reward và transition variance vẫn tồn tại
- Chạy cả `is_slippery=False` và `is_slippery=True` như một phép đối chứng để tách ảnh hưởng của policy sampling khỏi transition stochasticity
- Đo trực tiếp variance của TD target/TD error theo seed và theo giai đoạn huấn luyện; reward curve chỉ là bằng chứng bổ trợ

**4.3. Môi trường 3 — Taxi (`Taxi-v3`)**
*Trục kiểm tra: Sample efficiency và sparse reward*
*Cặp được phân biệt: Q-Learning vs Expected SARSA*

- Không gian: $|S| = 500$, $|A| = 6$, sparse reward (+20 thành công, -1 mỗi bước, -10 pickup/dropoff sai)
- Behavior policy quyết định mức exploration; Q-Learning không mặc nhiên exploration mạnh hơn Expected SARSA khi dùng cùng lịch $\epsilon$
- **Giả thuyết:** Q-Learning có thể đạt thành công sớm hơn nhờ target greedy, trong khi Expected SARSA có thể ổn định hơn; cả hai chiều phải được kiểm chứng thay vì xem là kết luận trước thí nghiệm
- Metric chính: **environment steps to first success**, **steps to threshold**, final evaluation return và reward variability across seeds

**4.4. Thiết lập siêu tham số**

| Tham số | Giá trị | Lý do |
|---|---|---|
| $\gamma$ | 0.99 | Đánh giá cao reward dài hạn |
| $\alpha$ | 0.1 (mặc định) | Ổn định; thay đổi trong sensitivity analysis |
| $\epsilon$ (Cliff) | 0.1 cố định | Để giữ hiện tượng On vs Off-policy |
| $\epsilon$ (Frozen, Taxi) | Decay từ 1.0 → 0.01 | Lịch thực nghiệm thực dụng; không được gọi là GLIE vì giới hạn dưới khác 0 |
| Số seeds | Tối thiểu 20, ưu tiên 30 seeds độc lập | Ước lượng uncertainty tốt hơn cho reward nhị phân và first-success time |
| Ngân sách huấn luyện | 500 (Cliff), 5000 (Frozen), 10000 (Taxi) episodes, đồng thời ghi số environment steps | So sánh cả theo episode và interaction budget |

**4.5. Quy trình kiểm soát thực nghiệm**

- Dùng cùng tập seed, khởi tạo Q-table, lịch $\alpha$, lịch $\epsilon$ và giới hạn bước cho các thuật toán trong cùng môi trường
- Quy định rõ cách xử lý `terminated` và `truncated`; không bootstrap qua terminal transition
- Tách **online training return** của behavior policy khỏi **evaluation return** của greedy policy. Evaluation chạy định kỳ, không exploration và không cập nhật Q
- Dùng random tie-breaking cho các action đồng hạng và định nghĩa chính xác xác suất $\epsilon$-greedy khi có nhiều greedy action
- Lưu raw result, cấu hình và seed để bảo đảm khả năng tái lập

---

## 5. BỘ CHỈ SỐ ĐÁNH GIÁ VÀ KẾT QUẢ THỰC NGHIỆM

**5.1. Bộ metrics hoàn chỉnh**

| # | Metric | Định nghĩa chính xác | Môi trường |
|---|---|---|---|
| M1 | Moving avg reward | Trung bình reward trên cửa sổ 50 episodes | Cả 3 |
| M2 | Steps to threshold | Số environment steps đầu tiên mà evaluation moving average vượt $\theta$ và duy trì trong các lần evaluation tiếp theo | Cả 3 |
| M3 | Performance uncertainty | Mean, standard deviation và 95% confidence interval của final evaluation return across seeds | Cả 3 |
| M4 | Steps to first success | Tổng số bước (không phải episode) trước khi agent lần đầu nhận reward dương | Taxi, Frozen Lake |
| M5 | Policy path visualization | Hình ảnh đường đi của greedy policy sau training | Cliff Walking |
| M6 | Q-value heatmap | Heatmap $\max_a Q(s,a)$ trên lưới sau training | Frozen Lake |
| M7 | Learning-curve AUC | Diện tích dưới evaluation learning curve theo environment steps | Cả 3 |
| M8 | TD-target variance | Variance của TD target/TD error trong các cửa sổ huấn luyện tương ứng | Frozen Lake, có thể báo cáo thêm cho Taxi |
| M9 | Safety/error rate | Tỷ lệ rơi vách hoặc hành động pickup/dropoff sai trong lúc training/evaluation | Cliff Walking, Taxi |

> **Lưu ý M2:** Ngưỡng $\theta$ phải được xác định trước khi xem kết quả và áp dụng trên evaluation policy. Có thể khởi đầu với Cliff Walking $\theta=-20$, Frozen Lake $\theta=0.7$ win rate và Taxi $\theta=5$, sau đó kiểm tra tính hợp lệ bằng baseline random và nghiệm gần tối ưu. Báo thêm tỷ lệ seed đạt ngưỡng để tránh che khuất các run không hội tụ.

**5.2. Phân tích kết quả và thảo luận**

*Cliff Walking:*
- Kiểm định giả thuyết Q-Learning có online training return thấp hơn và cliff-fall rate cao hơn khi $\epsilon$ cố định, dù greedy evaluation policy có thể chọn đường ngắn hơn
- Dùng policy path (M5) để kiểm tra liệu Q-Learning đi sát vách còn SARSA chọn đường an toàn hơn
- Liên hệ lý thuyết: $\max_a$ operator học đúng optimal value nhưng behavior policy $\epsilon$-greedy thực thi không theo đúng greedy → gap giữa target policy và behavior policy

*Frozen Lake:*
- So sánh reward std (M3) và Q-value heatmap (M6) giữa Expected SARSA và SARSA
- Dùng M8 để kiểm tra trực tiếp Expected SARSA có TD-target variance thấp hơn hay không trong cả hai cấu hình slippery và non-slippery
- Heatmap chỉ dùng để trực quan hóa, không dùng khái niệm "mượt hơn" làm bằng chứng về độ chính xác. Nếu cần đánh giá accuracy, so sánh với $V^*$ tính bằng Dynamic Programming hoặc báo Bellman residual

*Taxi-v3:*
- So sánh steps to first success (M4) và convergence episode (M2) giữa Q-Learning và Expected SARSA
- Kiểm tra liệu Q-Learning có M4/M2 thấp hơn hay không và liệu Expected SARSA có uncertainty thấp hơn hay không; chấp nhận kết quả bác bỏ giả thuyết
- Đo riêng wall-clock target-update time nếu thảo luận chi phí tính toán; về độ phức tạp, cả `max` và expectation đều quét qua 6 action trong triển khai tabular thông thường
- Nếu muốn kết luận về overestimation, cần đo sai lệch $\max_a Q(s,a)-V^*(s)$ trên các trạng thái kiểm soát được. Taxi chỉ là thí nghiệm sample efficiency, không tự nó chứng minh maximization bias

**5.3. Phân tích độ nhạy siêu tham số (Sensitivity Analysis)**

*Tác động của $\alpha$:*
- Thử $\alpha \in \{0.01, 0.1, 0.3, 0.5, 0.8\}$ trên Cliff Walking
- Quan sát cần kiểm chứng: $\alpha$ lớn có thể gây dao động hoặc hiệu năng không ổn định; trong tabular setting không nên mặc định gọi hiện tượng này là divergence. $\alpha$ nhỏ thường học chậm hơn dưới cùng ngân sách
- Liên hệ điều kiện Robbins-Monro: $\alpha$ cố định không thỏa điều kiện hội tụ tiệm cận; nó vẫn thường được dùng để theo dõi bài toán non-stationary hoặc đạt nghiệm gần ổn định trong ngân sách hữu hạn

*Tác động của $\epsilon$ cố định vs decay:*
- So sánh hai chế độ $\epsilon$ trên Cliff Walking để minh họa lại hiện tượng On/Off-policy đã phân tích ở 5.2
- Kiểm tra giả thuyết gap giữa SARSA và Q-Learning thu hẹp khi $\epsilon$ giảm. Chỉ liên hệ với GLIE nếu lịch decay thực sự tiến về 0 và vẫn bảo đảm infinite exploration; lịch có `epsilon_min=0.01` không đủ cho kết luận này

---

## 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

**6.1. Tổng kết**

| Thuật toán | Ưu điểm | Nhược điểm | Phù hợp khi |
|---|---|---|---|
| SARSA | Target phản ánh behavior policy; có thể học hành vi an toàn hơn khi exploration còn tồn tại | Có variance do sampling action kế tiếp; kết quả phụ thuộc exploration schedule | Môi trường có chi phí lỗi cao trong quá trình học |
| Q-Learning | Học target greedy độc lập behavior policy; có bảo đảm hội tụ tabular dưới các điều kiện chuẩn | Có thể có maximization bias và policy mismatch trong lúc training | Cần greedy target và có thể chấp nhận rủi ro exploration |
| Expected SARSA | Loại bỏ variance do sampling action kế tiếp; linh hoạt on/off-policy | Phải tính expectation qua action và vẫn chịu transition/reward variance | Action space nhỏ hoặc cần TD target ổn định hơn |

Không có thuật toán vượt trội tuyệt đối. Kết luận cuối cùng phải dựa trên kết quả cùng confidence interval, phân biệt rõ training behavior với greedy evaluation, và nêu các giả thuyết bị dữ liệu bác bỏ. Lựa chọn thuật toán phụ thuộc vào stochasticity, chi phí lỗi trong lúc học, target policy, action-space size và ngân sách tương tác.

**6.2. Hướng phát triển**
- Mở rộng sang Function Approximation (Semi-gradient SARSA, DQN) để xử lý không gian trạng thái liên tục
- Thảo luận về "Bộ ba chết chóc" (The Deadly Triad): Function approximation + Bootstrapping + Off-policy → instability
- Double Q-Learning như giải pháp cho maximization bias của Q-Learning

---
