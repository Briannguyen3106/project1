# 4. Thiết kế thực nghiệm

## 4.1. Nguyên tắc thiết kế

Mỗi môi trường trong nghiên cứu được lựa chọn để trả lời một câu hỏi cụ thể, không phải để tạo một bảng xếp hạng tổng quát. Các thuật toán trong cùng một phép so sánh dùng chung ngân sách environment steps, tập seed, Q initialization, policy exploration và quy tắc đánh giá. Biến độc lập chính là bootstrap target.

Mọi nhận định về thuật toán trước khi chạy code được trình bày dưới dạng giả thuyết. Nếu dữ liệu không ủng hộ giả thuyết, báo cáo phải giữ kết quả đó và phân tích nguyên nhân thay vì điều chỉnh metric sau khi quan sát.

## 4.2. Thí nghiệm 1: Cliff Walking

### 4.2.1. Môi trường

`CliffWalking-v0` là lưới $4\times12$. Tác tử bắt đầu ở góc dưới bên trái và phải tới đích ở góc dưới bên phải. Các ô giữa điểm đầu và đích trên hàng cuối là vách. Một bước thông thường nhận reward $-1$; rơi vách nhận penalty lớn và đưa tác tử trở lại điểm đầu theo định nghĩa môi trường.

Môi trường này phù hợp để khảo sát khác biệt on-policy/off-policy về an toàn. Đường đi ngắn nhất chạy sát vách. Khi behavior policy vẫn có exploration, chỉ một action ngẫu nhiên không phù hợp cũng có thể gây penalty lớn.

### 4.2.2. Thuật toán và giả thuyết

So sánh chính: SARSA và Q-Learning với cùng policy $\epsilon$-greedy, $\epsilon=0.1$ cố định.

SARSA bootstrap theo action tiếp theo thực sự được behavior policy chọn. Vì vậy, giá trị của đường sát vách bao gồm rủi ro exploration. Q-Learning bootstrap theo greedy action và không đưa action khám phá tiếp theo vào target. Giả thuyết là SARSA học đường an toàn hơn, có cliff-fall rate thấp hơn và online return tốt hơn; Q-Learning có greedy path ngắn hơn khi evaluation không exploration.

### 4.2.3. Biến đo

- Online training return theo episode và theo environment steps.
- Greedy evaluation return định kỳ.
- Số và tỷ lệ rơi vách.
- Độ dài greedy path từ start tới goal.
- Learning-curve AUC.
- Tỷ lệ seed đạt threshold.

### 4.2.4. Kiểm soát

Giữ $\epsilon$ cố định trong thí nghiệm chính để duy trì policy mismatch. Thí nghiệm sensitivity mới thay bằng epsilon decay. Việc đánh giá greedy policy phải chạy ở episode riêng, không cập nhật Q và không cộng vào training budget.

## 4.3. Thí nghiệm 2: Frozen Lake

### 4.3.1. Môi trường

`FrozenLake-v1` mô tả bài toán điều hướng trên hồ đóng băng. Tác tử phải đi từ điểm xuất phát tới goal mà không rơi vào hole. Reward thường bằng 1 khi tới goal và 0 ở các trường hợp khác, nên tín hiệu học thưa và biến kết quả có dạng gần Bernoulli.

Với `is_slippery=True`, action dự định không quyết định hoàn toàn hướng di chuyển. Transition stochasticity làm $S_{t+1}$ ngẫu nhiên. Với `is_slippery=False`, transition được xác định bởi action. Hai cấu hình được dùng như đối chứng.

### 4.3.2. Thuật toán và giả thuyết

So sánh chính: SARSA và Expected SARSA on-policy, cùng behavior policy $\epsilon$-greedy.

Hai thuật toán có cùng kỳ vọng theo action nếu Q-table và policy giống nhau, nhưng SARSA dùng một action sample còn Expected SARSA tính trung bình đầy đủ. Giả thuyết là Expected SARSA có TD-target variance thấp hơn. Chênh lệch trong cấu hình non-slippery giúp phản ánh action-sampling variance; cấu hình slippery cho thấy transition variance vẫn tồn tại.

### 4.3.3. Biến đo

- Success rate của greedy evaluation policy.
- Online return và evaluation return.
- Variance của TD target và TD error theo cửa sổ environment steps.
- Uncertainty across seeds.
- Steps to first success và steps to threshold.
- Bellman residual hoặc sai số so với $V^*$ nếu tính được bằng Dynamic Programming.
- Heatmap $\max_aQ(s,a)$ để minh họa, không dùng làm bằng chứng định lượng độc lập.

### 4.3.4. Phân tích đối chứng

So sánh thuật toán trong từng cấu hình trước, sau đó mới so sánh mức thay đổi từ non-slippery sang slippery. Không kết luận Expected SARSA loại bỏ toàn bộ variance. Nếu reward variance giữa seed không giảm dù TD-target variance giảm, cần giải thích rằng variance cập nhật cục bộ và variance của hiệu năng cuối là hai đại lượng khác nhau.

## 4.4. Thí nghiệm 3: Taxi

### 4.4.1. Môi trường

`Taxi-v3` có 500 trạng thái rời rạc và 6 hành động. Tác tử phải đến vị trí hành khách, đón đúng cách, di chuyển đến đích và trả khách. Reward gồm penalty theo bước, penalty lớn cho pickup/dropoff sai và reward dương khi hoàn thành nhiệm vụ.

Không gian lớn hơn Cliff Walking và Frozen Lake tạo bài toán khám phá dài hơn. Đây là môi trường phù hợp để đo sample efficiency, lỗi hành động và độ ổn định qua nhiều seed.

### 4.4.2. Thuật toán và giả thuyết

So sánh chính: Q-Learning và Expected SARSA on-policy với cùng behavior policy. Q-Learning dùng greedy target; Expected SARSA dùng trung bình theo $\epsilon$-greedy policy.

Giả thuyết là Q-Learning có thể đạt first success hoặc threshold sớm hơn, trong khi Expected SARSA có thể có uncertainty thấp hơn. Không gọi Q-Learning là "aggressive exploration", vì exploration do behavior policy quyết định. Taxi cũng không được dùng như bằng chứng trực tiếp của maximization bias nếu không có giá trị tham chiếu.

### 4.4.3. Biến đo

- Environment steps to first successful episode.
- Environment steps to evaluation threshold.
- Final greedy evaluation return.
- Evaluation success rate.
- Tỷ lệ pickup/dropoff sai.
- Learning-curve AUC.
- Mean, standard deviation và 95% confidence interval across seeds.
- Wall-clock update time tách khỏi environment interaction time.

## 4.5. Siêu tham số dự kiến

| Thành phần | Cliff Walking | Frozen Lake | Taxi |
|---|---:|---:|---:|
| $\gamma$ | 0.9 | 0.9 | 0.9 |
| $\alpha$ mặc định | 0.1 | 0.1 | 0.1 |
| $\epsilon$ | 0.1 cố định | 1.0 giảm đến 0.01 | 1.0 giảm đến 0.01 |
| Episode budget dự kiến | 500 | 5000 | 10000 |
| Seed | tối thiểu 20, ưu tiên 30 | tối thiểu 20, ưu tiên 30 | tối thiểu 20, ưu tiên 30 |

Episode budget chỉ là giới hạn triển khai ban đầu. Tất cả learning curve phải đồng thời lưu cumulative environment steps để so sánh công bằng khi độ dài episode khác nhau.

Lịch epsilon giảm tới 0.01 là lịch thực nghiệm hữu hạn, không được mô tả là GLIE. Nếu cần thí nghiệm minh họa GLIE, phải dùng một schedule tiến về 0 và thảo luận riêng điều kiện infinite exploration.

## 4.6. Quy trình một run

1. Khởi tạo môi trường, RNG và Q-table bằng seed được chỉ định.
2. Huấn luyện theo algorithm và configuration cố định.
3. Ghi từng transition cần thiết cho reward, TD target, TD error, safety event và cumulative steps.
4. Sau mỗi khoảng cố định theo environment steps, chạy nhiều evaluation episode bằng greedy policy.
5. Không cập nhật Q, không dùng exploration và không tính evaluation transition vào training budget.
6. Lưu raw data cùng metadata: algorithm, environment, seed, hyperparameter và phiên bản thư viện.
7. Lặp lại trên toàn bộ seed set giống nhau cho các thuật toán.

## 4.7. Xử lý terminal và truncation

Nếu `terminated=True`, target chỉ chứa reward:

$$
Y_t=R_{t+1}.
$$

Nếu `truncated=True` do giới hạn bước, implementation phải ghi rõ có bootstrap hay không theo cách diễn giải task. Quyết định này phải giống nhau giữa các thuật toán. Episode cap cần đủ lớn để không biến truncation thành nguồn sai lệch chính.

## 4.8. Sensitivity analysis

### Learning rate

Thử $\alpha\in\{0.01,0.1,0.3,0.5,0.8\}$ trên Cliff Walking. Mục tiêu là đánh giá tốc độ học, dao động và final performance. Không mặc định gọi hiệu năng xấu với $\alpha$ lớn là divergence; divergence chỉ nên dùng khi ước lượng tăng không bị chặn hoặc có bằng chứng rõ.

### Exploration schedule

So sánh $\epsilon=0.1$ cố định với epsilon decay trên Cliff Walking. Giả thuyết là khoảng cách về online safety giữa SARSA và Q-Learning thu hẹp khi exploration giảm. Kết quả phải trình bày cùng cliff-fall rate và greedy evaluation để tránh nhầm giữa giảm exploration và cải thiện value estimate.

## 4.9. Khả năng tái lập

Mã nguồn cần xuất một file cấu hình cho mỗi experiment batch. Raw result không được chỉ lưu dưới dạng hình. Mỗi hàng dữ liệu tối thiểu gồm environment, algorithm, seed, episode, cumulative step, return, episode length, epsilon và các event đặc thù. Phiên bản Python, Gymnasium, NumPy và hệ điều hành cần được ghi lại trong phụ lục hoặc metadata.
