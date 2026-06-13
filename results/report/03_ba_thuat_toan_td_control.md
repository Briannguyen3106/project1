# 3. Phân tích ba thuật toán TD Control

## 3.1. Khung cập nhật chung

Cả ba thuật toán đều duy trì bảng $Q(s,a)$ và cập nhật cặp vừa trải nghiệm theo dạng:

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha\left[Y_t-Q(S_t,A_t)\right],
$$

trong đó $Y_t$ là TD target. Vì reward, transition và policy được giữ giống nhau trong một phép so sánh có kiểm soát, khác biệt hành vi chủ yếu đến từ cách xác định phần bootstrap của $Y_t$.

## 3.2. SARSA

### 3.2.1. Công thức cập nhật

SARSA nhận tên từ tuple gồm năm thành phần $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$. Target và TD error là:

$$
Y_t^{\text{SARSA}}
=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),
$$

$$
\delta_t^{\text{SARSA}}
=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t).
$$

Action $A_{t+1}$ được lấy từ chính policy đang tạo dữ liệu. Do đó SARSA là on-policy: thuật toán đánh giá hệ quả của policy mà nó thực sự thực thi, bao gồm cả xác suất chọn action khám phá.

### 3.2.2. Quy trình

1. Khởi tạo $Q(s,a)$, thường bằng 0, và đặt giá trị terminal bằng 0.
2. Ở đầu episode, chọn $A_0$ từ behavior policy dựa trên $Q(S_0,\cdot)$.
3. Thực hiện $A_t$, nhận $R_{t+1}$ và $S_{t+1}$.
4. Nếu transition chưa terminal, chọn $A_{t+1}$ từ cùng behavior policy.
5. Cập nhật $Q(S_t,A_t)$ bằng target chứa $Q(S_{t+1},A_{t+1})$.
6. Gán $(S_t,A_t)\leftarrow(S_{t+1},A_{t+1})$ và lặp lại.

Điểm cần chú ý là action kế tiếp được chọn trước khi cập nhật vòng lặp tiếp theo. Nếu implementation chọn một action để tính target nhưng sau đó lại chọn action khác để tương tác, thuật toán không còn đúng với SARSA chuẩn.

### 3.2.3. Đặc điểm

SARSA phản ánh rủi ro của behavior policy trong target. Nếu một trạng thái nằm gần vùng nguy hiểm và $\epsilon>0$, giá trị của trạng thái đó bao gồm khả năng action khám phá gây hậu quả xấu. Vì vậy, SARSA có thể ưu tiên đường đi dài hơn nhưng ít nhạy với lỗi exploration hơn.

Đổi lại, target sử dụng một action sample nên có thêm variance. Hai lần đến cùng một $S_{t+1}$ có thể tạo target khác nhau chỉ vì $A_{t+1}$ khác nhau. Với $\epsilon$ cố định, SARSA tiếp tục học giá trị liên quan tới policy $\epsilon$-soft đang được cải thiện. Với GLIE, Robbins-Monro và các điều kiện tabular chuẩn, SARSA có thể hội tụ về $Q^*$.

## 3.3. Q-Learning

### 3.3.1. Công thức cập nhật

Q-Learning dùng greedy bootstrap target:

$$
Y_t^{\text{Q}}
=R_{t+1}+\gamma\max_a Q(S_{t+1},a),
$$

$$
\delta_t^{\text{Q}}
=R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t).
$$

Behavior policy có thể là $\epsilon$-greedy để bảo đảm exploration, nhưng target policy là greedy đối với Q-table hiện tại. Vì target policy và behavior policy khác nhau, Q-Learning là off-policy.

### 3.3.2. Quy trình

1. Khởi tạo $Q(s,a)$ và policy khám phá.
2. Chọn $A_t$ từ behavior policy.
3. Thực hiện action và quan sát transition.
4. Nếu chưa terminal, tính giá trị lớn nhất trong hàng $Q(S_{t+1},\cdot)$.
5. Cập nhật cặp $(S_t,A_t)$.
6. Chọn action behavior mới và tiếp tục.

Action thực sự được chọn ở trạng thái kế tiếp không xuất hiện trong target. Chính sự tách biệt này cho phép Q-Learning học greedy target từ dữ liệu được tạo bởi một policy khác.

### 3.3.3. Đặc điểm

Trong tabular MDP hữu hạn, Q-Learning có bảo đảm hội tụ về $Q^*$ khi mọi cặp $(s,a)$ được thăm vô hạn lần, reward bị chặn và learning rate thỏa các điều kiện stochastic approximation thích hợp.

Tuy nhiên, greedy target có hai hệ quả thực nghiệm. Thứ nhất, target không phản ánh trực tiếp rủi ro của action khám phá tiếp theo. Trong Cliff Walking, điều này có thể khiến greedy policy đi sát vách dù behavior policy vẫn có xác suất chọn nhầm hướng. Thứ hai, phép cực đại trên các ước lượng nhiễu có thể tạo maximization bias. Cần phân biệt hai hiện tượng: policy mismatch liên quan tới khác biệt giữa target và behavior policy; maximization bias liên quan tới thống kê của toán tử max.

## 3.4. Expected SARSA

### 3.4.1. Công thức cập nhật

Expected SARSA thay action sample bằng kỳ vọng theo một target policy $\pi$:

$$
Y_t^{\text{Expected}}
=R_{t+1}
+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a),
$$

$$
\delta_t^{\text{Expected}}
=Y_t^{\text{Expected}}-Q(S_t,A_t).
$$

Trong cấu hình chính của báo cáo, $\pi$ là behavior policy $\epsilon$-greedy, nên Expected SARSA được dùng theo kiểu on-policy. Nếu expectation được tính theo một policy khác behavior policy, thuật toán có thể hoạt động off-policy.

### 3.4.2. Kỳ vọng dưới policy $ε$-greedy

Với tập greedy action $\mathcal{G}(s)$, xác suất policy nên được tính chính xác kể cả khi có tie:

$$
\pi(a\mid s)
=\frac{\epsilon}{|\mathcal{A}|}
+\mathbb{I}[a\in\mathcal{G}(s)]\frac{1-\epsilon}{|\mathcal{G}(s)|}.
$$

Kỳ vọng bootstrap là tích vô hướng giữa vector xác suất action và hàng Q-value. Cách tính này tránh sai lệch do giả định chỉ có một greedy action.

### 3.4.3. Đặc điểm

Conditional trên $S_{t+1}$, Expected SARSA tính trung bình chính xác qua mọi action thay vì chọn một action ngẫu nhiên. Vì vậy, nó loại bỏ thành phần variance do sampling $A_{t+1}$. Điều này thường cho phép sử dụng learning rate lớn hơn SARSA trong một số bài toán, nhưng kết luận cụ thể vẫn phụ thuộc môi trường.

Expected SARSA không loại bỏ transition hoặc reward variance. Nó cũng không mặc nhiên không có bias, bởi Q-table dùng trong bootstrap vẫn là ước lượng. Khi target policy hoàn toàn greedy, biểu thức expectation thu về $\max_aQ(S_{t+1},a)$ và thuật toán tương đương Q-Learning về target.

## 3.5. Quan hệ giữa ba thuật toán

Ba thuật toán có thể được nhìn như ba cách xử lý biến action kế tiếp:

- SARSA lấy một mẫu từ policy.
- Expected SARSA lấy kỳ vọng đầy đủ theo policy.
- Q-Learning lấy giá trị của greedy action, tương đương expectation theo deterministic greedy target policy.

SARSA và Expected SARSA on-policy có cùng kỳ vọng target khi điều kiện trên transition và Q-table hiện tại, nhưng khác variance do action sampling. Q-Learning và Expected SARSA greedy có cùng target. Vì vậy, Expected SARSA đóng vai trò cầu nối khái niệm giữa sample backup và max backup.

## 3.6. Bảng so sánh

| Tiêu chí | SARSA | Q-Learning | Expected SARSA |
|---|---|---|---|
| Bootstrap | $Q(S',A')$ | $\max_a Q(S',a)$ | $\sum_a\pi(a\mid S')Q(S',a)$ |
| Policy chính | On-policy | Off-policy | On-policy hoặc off-policy |
| Sampling action trong target | Có | Không | Không |
| Variance do $A'$ | Có | Không | Không |
| Toán tử max | Không | Có | Chỉ khi target policy greedy |
| Policy mismatch | Không trong cấu hình chuẩn | Có | Tùy target policy |
| Chi phí tính bootstrap | $O(1)$ sau khi có $A'$ | $O(|A|)$ | $O(|A|)$ |
| Điểm mạnh | Phản ánh behavior risk | Hướng trực tiếp tới greedy target | Target kỳ vọng ít biến động hơn |
| Rủi ro chính | Action-sampling variance | Maximization bias và behavior-target gap | Chi phí expectation, vẫn còn transition variance |

## 3.7. Cài đặt công bằng

Để so sánh đúng, ba thuật toán phải dùng cùng Q initialization, seed set, behavior-policy schedule, discount factor, learning rate, episode cap và interaction budget. Q-Learning không được nhận nhiều exploration hơn chỉ vì nó là off-policy. Expected SARSA không được dùng target policy khác mà không công bố rõ.

Ngoài ra, thời gian tính toán nên được đo riêng với sample efficiency. Q-Learning và Expected SARSA đều thường quét action space để tính target, nên không nên ghi Q-Learning là $O(1)$ trong một implementation tabular thông thường.
