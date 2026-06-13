# 2. Cơ sở lý thuyết

## 2.1. Quá trình quyết định Markov

Một bài toán học tăng cường thường được mô hình hóa bằng quá trình quyết định Markov (Markov Decision Process, MDP):

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,\gamma),
$$

trong đó $\mathcal{S}$ là tập trạng thái, $\mathcal{A}$ là tập hành động, $p(s',r\mid s,a)$ là phân phối xác suất của trạng thái và phần thưởng kế tiếp, còn $\gamma\in[0,1]$ là hệ số chiết khấu.

Tính chất Markov phát biểu rằng khi biết trạng thái hiện tại và hành động hiện tại, phân phối của kết quả kế tiếp không phụ thuộc thêm vào toàn bộ lịch sử:

$$
\Pr(S_{t+1},R_{t+1}\mid S_t,A_t,S_{t-1},A_{t-1},\ldots)
=
\Pr(S_{t+1},R_{t+1}\mid S_t,A_t).
$$

Policy $\pi(a\mid s)$ là phân phối xác suất chọn hành động $a$ tại trạng thái $s$. Nếu policy gán xác suất 1 cho một hành động tại mỗi trạng thái thì đó là deterministic policy; ngược lại là stochastic policy. Trong các thí nghiệm của báo cáo, behavior policy chủ yếu là $\epsilon$-greedy.

## 2.2. Return và hệ số chiết khấu

Mục tiêu của tác tử là tối đa hóa kỳ vọng của return. Với bài toán episodic, return từ thời điểm $t$ được định nghĩa:

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

Hệ số $\gamma$ điều chỉnh mức quan trọng của phần thưởng tương lai. Khi $\gamma$ gần 0, tác tử ưu tiên phần thưởng tức thời. Khi $\gamma$ gần 1, tác tử xem trọng hệ quả dài hạn hơn. Trong episodic task hữu hạn, $\gamma=1$ có thể hợp lệ; tuy nhiên báo cáo sử dụng $\gamma=0.99$ để nhất quán giữa các môi trường và giữ toán tử Bellman có tính co trong các lập luận discounted setting.

## 2.3. Hàm giá trị trạng thái và hành động

Giá trị trạng thái dưới policy $\pi$ là kỳ vọng return khi bắt đầu ở trạng thái $s$ và tiếp tục theo $\pi$:

$$
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
$$

Giá trị hành động là:

$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

Trong control, hàm giá trị tối ưu được định nghĩa:

$$
q_*(s,a)=\max_\pi q_\pi(s,a).
$$

Nếu biết $q_*$, một optimal policy có thể được xây dựng bằng cách chọn action đạt cực đại tại mỗi trạng thái. Trong thực tế, các thuật toán TD control duy trì ước lượng $Q(s,a)$ và đồng thời dùng nó để cải thiện policy.

## 2.4. Phương trình Bellman

Bellman expectation equation phân rã giá trị hiện tại thành phần thưởng một bước và giá trị kỳ vọng của trạng thái kế tiếp:

$$
q_\pi(s,a)
=
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].
$$

Bellman optimality equation thay expectation theo policy bằng phép cực đại:

$$
q_*(s,a)
=
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}q_*(s',a')\right].
$$

Đặt toán tử Bellman optimality $\mathcal{T}_*$ tác động lên một hàm $Q$:

$$
(\mathcal{T}_*Q)(s,a)
=
\mathbb{E}\left[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')\mid S_t=s,A_t=a\right].
$$

Với $\gamma<1$, toán tử này là contraction theo chuẩn cực đại:

$$
\|\mathcal{T}_*Q_1-\mathcal{T}_*Q_2\|_\infty
\leq
\gamma\|Q_1-Q_2\|_\infty.
$$

Do đó, $\mathcal{T}_*$ có một điểm cố định duy nhất là $Q^*$. Kết quả này là nền tảng cho Dynamic Programming và cho các chứng minh hội tụ stochastic approximation của Q-Learning. Tuy nhiên, tính co của toán tử kỳ vọng không tự động bảo đảm một thuật toán lấy mẫu hữu hạn hội tụ; còn cần điều kiện về step-size và exploration.

## 2.5. Dynamic Programming, Monte Carlo và Temporal Difference

Dynamic Programming thực hiện backup dựa trên kỳ vọng đầy đủ qua mô hình môi trường. Ưu điểm là cập nhật có cấu trúc và có các bảo đảm rõ trong MDP hữu hạn, nhưng yêu cầu biết $p(s',r\mid s,a)$ và phải quét qua nhiều trạng thái.

Monte Carlo không cần mô hình. Thuật toán sử dụng return quan sát được $G_t$ làm target:

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)].
$$

MC target không bootstrap nên không chịu bias từ ước lượng giá trị kế tiếp, nhưng thường có variance cao và chỉ được xác định sau khi episode kết thúc.

TD(0) dùng target một bước:

$$
V(S_t)\leftarrow V(S_t)+\alpha
\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right].
$$

TD vừa lấy mẫu transition từ môi trường vừa bootstrap từ ước lượng hiện tại. Nó có thể học online sau mỗi bước và không cần mô hình. Đổi lại, target phụ thuộc vào một giá trị chưa chính xác, tạo bias hữu hạn mẫu. Sự đánh đổi bias-variance này là đặc điểm quan trọng của họ TD.

## 2.6. TD target và TD error

Với state-value prediction, TD target là:

$$
Y_t=R_{t+1}+\gamma V(S_{t+1}),
$$

và TD error là:

$$
\delta_t=Y_t-V(S_t)
=R_{t+1}+\gamma V(S_{t+1})-V(S_t).
$$

TD error biểu diễn độ bất ngờ của transition so với dự đoán hiện tại. Nếu target lớn hơn giá trị đang lưu, cập nhật tăng ước lượng; nếu nhỏ hơn, cập nhật giảm ước lượng. Trong action-value control, cấu trúc vẫn tương tự nhưng thành phần bootstrap phụ thuộc thuật toán.

Ở terminal transition, không tồn tại giá trị tương lai. Vì vậy target phải được xử lý thành $Y_t=R_{t+1}$, tương đương đặt giá trị bootstrap bằng 0. `truncated` do giới hạn thời gian cần được xử lý theo đúng ngữ nghĩa môi trường; không nên tự động đồng nhất mọi truncation với trạng thái terminal tự nhiên mà không ghi rõ quy ước.

## 2.7. Policy $ε$-greedy

Policy $\epsilon$-greedy cân bằng exploitation và exploration. Với một greedy action duy nhất $a^*$ trong $|\mathcal{A}|$ hành động:

$$
\pi(a\mid s)=
\begin{cases}
1-\epsilon+\dfrac{\epsilon}{|\mathcal{A}|}, & a=a^*,\\[6pt]
\dfrac{\epsilon}{|\mathcal{A}|}, & a\neq a^*.
\end{cases}
$$

Nếu có nhiều action cùng đạt giá trị cực đại, phần xác suất khai thác cần được chia đều giữa các greedy action. Random tie-breaking tránh thiên lệch cố định về action có chỉ số nhỏ nhất. Chi tiết này đặc biệt quan trọng khi Q-table ban đầu chứa cùng một giá trị cho mọi action.

Behavior policy là policy thực sự tạo dữ liệu. Target policy là policy mà phép cập nhật đang đánh giá hoặc hướng tới. Với on-policy learning, hai policy này giống nhau. Với off-policy learning, chúng có thể khác nhau.

## 2.8. Stochastic approximation và điều kiện hội tụ

Các cập nhật TD có dạng tổng quát:

$$
Q_{t+1}(S_t,A_t)
=Q_t(S_t,A_t)+\alpha_t(S_t,A_t)\delta_t.
$$

Điều kiện Robbins-Monro thường được phát biểu riêng cho mỗi cặp $(s,a)$:

$$
\sum_{t=0}^{\infty}\alpha_t(s,a)=\infty,
\qquad
\sum_{t=0}^{\infty}\alpha_t^2(s,a)<\infty.
$$

Điều kiện thứ nhất bảo đảm thuật toán không ngừng học quá sớm; điều kiện thứ hai làm ảnh hưởng của nhiễu giảm dần. Ví dụ, learning rate theo số lần thăm $\alpha_n=1/n$ thỏa cả hai điều kiện. Learning rate cố định không thỏa điều kiện thứ hai, dù vẫn thường hữu ích trong thực nghiệm hữu hạn hoặc môi trường non-stationary.

Robbins-Monro không phải điều kiện đủ độc lập. Hội tụ còn phụ thuộc MDP hữu hạn, bounded reward, visitation và cấu trúc toán tử cập nhật. Đối với control, exploration phải bảo đảm mọi cặp trạng thái-hành động liên quan được cập nhật đủ nhiều.

## 2.9. GLIE

GLIE là viết tắt của Greedy in the Limit with Infinite Exploration. Một chuỗi policy thỏa GLIE khi:

1. Mỗi cặp $(s,a)$ được chọn vô hạn lần.
2. Xác suất chọn action không greedy tiến về 0 khi thời gian tiến tới vô hạn.

Điều kiện thứ nhất duy trì khả năng sửa các ước lượng sai; điều kiện thứ hai làm behavior policy tiến dần về greedy policy. Một lịch giảm $\epsilon$ đến 0.01 rồi giữ cố định không phải GLIE, vì policy không greedy in the limit. Trong báo cáo, cần phân biệt lịch decay thực dụng dùng cho ngân sách hữu hạn với lịch được dùng trong phát biểu hội tụ tiệm cận.

## 2.10. Bias và variance của TD target

TD target là biến ngẫu nhiên do nhiều nguồn: reward, transition kế tiếp và action kế tiếp. Theo định luật total variance, với trạng thái kế tiếp đã biết, variance do action sampling có thể được tách khỏi các nguồn còn lại. SARSA lấy một $A_{t+1}$ nên giữ thành phần này. Expected SARSA tính expectation theo $\pi$, do đó loại bỏ conditional variance phát sinh riêng từ việc lấy mẫu $A_{t+1}$.

Expected SARSA không loại bỏ variance của $R_{t+1}$ hoặc $S_{t+1}$. Vì vậy, trong môi trường slippery, target vẫn biến động ngay cả khi expectation qua action được tính chính xác.

Q-Learning không lấy mẫu action kế tiếp trong target, nhưng dùng cực đại của các ước lượng nhiễu. Nói chung:

$$
\mathbb{E}\left[\max_a \hat{Q}(s,a)\right]
\geq
\max_a\mathbb{E}[\hat{Q}(s,a)].
$$

Bất đẳng thức này giải thích khả năng xuất hiện maximization bias. Mức bias phụ thuộc phân phối sai số, số action và tương quan giữa các ước lượng; không phải mọi bài toán stochastic đều nhất thiết biểu hiện bias đủ lớn để quan sát.
