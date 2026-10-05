# Báo cáo Day 19 — Flat RAG vs GraphRAG

**Họ tên:** Nguyễn Minh Hiếu  **MSSV:** 2A202602669  **Ngày:** 2026-10-05

> Kỳ vọng và thang điểm: `SUBMISSION.md`. Mọi số liệu phải khớp với `ket_qua_benchmark_kg.txt`. Bản thiết kế ontology nộp riêng ở `report/ONTOLOGY.md`.

## 1. Chi phí (10 điểm)

```
Chat model: gemini:gemini-3.1-flash-lite | Embedding: gemini:gemini-embedding-001 | top_k=3 | chunk_size=800 | chunks=176 | KG: 253 nodes / 548 rels

== Indexing (one-off)
pipeline  calls    in_tok  out_tok       USD  seconds
flat        176         0        0   0.00000    105.3
graph       196     39199     7188   0.00000    185.5

== Querying (mean per question)
pipeline  recall  judge   in_tok  out_tok       USD  seconds
flat        0.51   1.33      696       71   0.00000     3.97
graph       1.00   2.00     2236      157   0.00000     4.28
```

| Chỉ số | Flat | Graph | Graph / Flat |
| --- | --- | --- | --- |
| Indexing USD | 0.00000 | 0.00000 | không tính được (xem ghi chú) |
| Indexing giây | 105.3 | 185.5 | ×1.76 |
| Mỗi câu: USD | 0.00000 | 0.00000 | không tính được (xem ghi chú) |
| Mỗi câu: giây | 3.97 | 4.28 | ×1.08 |
| Mỗi câu: in_tok | 696 | 2236 | ×3.21 |

**Ghi chú về cột USD.** Model mặc định `gemini-2.5-flash-lite` không còn mở cho key mới, nên tôi chạy bằng `GEMINI_CHAT_MODEL=gemini-3.1-flash-lite`. Bảng giá trong `src/llm.py` không có model này và không có giá embedding của Gemini, nên script ghi 0 USD. API embedding của Gemini cũng không trả số token, nên `in_tok` của indexing Flat là 0. Vì vậy tôi so sánh bằng token và giây.

Nếu tạm lấy đơn giá của `gemini-2.5-flash-lite` trong `src/llm.py` (0,10 USD / 1 triệu token vào, 0,40 USD / 1 triệu token ra) làm **ước tính**, không phải số đo:

| Ước tính | Flat | Graph | Graph / Flat |
| --- | --- | --- | --- |
| Dựng graph (20 lần gọi LLM) | 0 | 39199 × 0,10 + 7188 × 0,40 ≈ 0,0068 USD | chi phí chỉ Graph có |
| Mỗi câu hỏi | 696 × 0,10 + 71 × 0,40 ≈ 0,000098 USD | 2236 × 0,10 + 157 × 0,40 ≈ 0,00029 USD | ×2,9 |

**Chi phí tăng thêm đến từ đâu?**

> Ở indexing, toàn bộ phần tăng là 20 lần gọi LLM để trích xuất 20 bài báo (39199 token vào, 7188 token ra, thêm khoảng 80 giây); phần luật trích bằng regex nên không tốn token. Ở mỗi câu hỏi, phần tăng là dữ kiện graph chèn vào prompt: 2236 so với 696 token vào, tức thêm khoảng 1540 token mỗi câu. Thời gian mỗi câu gần như không đổi (×1.08) vì truy vấn Cypher nhanh so với lần gọi LLM; số giây còn bị nhiễu bởi các lần chờ thử lại do giới hạn tốc độ của key miễn phí.
>
> GraphRAG không có điểm hòa vốn về tiền: nó đắt hơn ở cả lúc dựng lẫn mỗi câu hỏi. Thứ đổi lại là độ chính xác trên câu hỏi xuyên 2 KB (mục 2). Cách thiết kế `context()` ảnh hưởng lớn tới chi phí mỗi câu: với ontology gợi ý, cùng benchmark này tốn 5621 token vào mỗi câu (`ket_qua_benchmark_kg.hint.txt`), vì nó đưa nguyên văn cả khoản luật vào prompt.

## 2. Từng câu hỏi (10 điểm)

| Câu | Loại | Flat recall / judge | Graph recall / judge | Thắng | Vì sao (1 câu) |
| --- | --- | --- | --- | --- | --- |
| Q1 | single-hop-law | 1.00 / 2 | 1.00 / 2 | Hòa | Định nghĩa nằm gọn trong một chunk của Điều 2 Luật PCMT; graph không thêm gì. |
| Q2 | single-hop-news | 1.00 / 2 | 1.00 / 2 | Hòa | Hai tên bị cáo và mức án nằm trong cùng một bài báo. |
| Q3 | cross-kb | 0.33 / 1 | 1.00 / 2 | Graph | Flat chỉ lấy được chunk tin tức nên thiếu Điều 251 và khung 02–07 năm; graph đi `Person → Charge → Crime ← Article → Clause`. |
| Q4 | cross-kb | 0.33 / 1 | 1.00 / 2 | Graph | Flat biết hành vi nhưng không có Điều 255; graph lấy khoản `cao nhất` của Điều. |
| Q5 | cross-kb-multi-hop | 0.40 / 1 | 1.00 / 2 | Graph | Cần so 9,6 kg MDMA với ngưỡng trong luật; graph chọn sẵn khoản 4 điểm b qua `THRESHOLD`. |
| Q6 | aggregation | 0.00 / 1 | 1.00 / 2 | Graph | Flat chỉ thấy 3 chunk gần nhất (2 trong số đó cùng một vụ); graph liệt kê mọi `Case` có `INVOLVES` tới MDMA. |

Quy luật: câu trả lời nằm trong một đoạn văn (Q1, Q2) thì hai bên hòa và Flat rẻ hơn 3 lần về token. Câu cần ghép tin tức với luật (Q3–Q5) hoặc cần gom nhiều tài liệu (Q6) thì Flat thua ở cả 4 câu, vì top-3 chunk không bao giờ chứa đủ hai phía.

## 3. Phân tích lỗi (20 điểm)

Các truy vấn dưới đây chạy trên graph do `python bench_kg.py --judge` dựng (253 node / 548 cạnh).

### Lỗi E3: Trùng thực thể — một chuyên án vẫn thành hai `Case`

- **Hiện tượng:** Phan Kim Nhi (TikToker Phannhibeauty) bị bắt trong cùng chuyên án với Dương Minh Tuấn ("Hoàng Nato"), nhưng graph có hai `Case` cho chuyên án này và bà Nhi thuộc cả hai.
- **Bằng chứng:**

```cypher
MATCH (p:Person)-[:FACES]->(:Charge)-[:IN_CASE]->(k:Case)
WHERE p.key IN ['phan kim nhi', 'dương minh tuấn']
OPTIONAL MATCH (r:Report)-[:REPORTS]->(k)
RETURN p.name AS person, k.id AS case_id, k.name AS case, collect(DISTINCT r.doc_id) AS reports
ORDER BY person, case_id;
```

```
person           case_id                     case                            reports
Dương Minh Tuấn  news-100260920221957595#0   Dương Minh Tuấn (Hoàng Nato)    [news-100260925144412498, news-100260924095400982, news-100260922111804786, news-100260920221957595]
Phan Kim Nhi     news-100260920221957595#0   Dương Minh Tuấn (Hoàng Nato)    [news-100260925144412498, news-100260924095400982, news-100260922111804786, news-100260920221957595]
Phan Kim Nhi     news-100260922111804786#0   Phan Kim Nhi (Phannhibeauty)    [news-100260927182621527, news-100260922111804786]
```

- **Nguyên nhân:** Nằm ở hai bước. (1) Prompt trích xuất: với bài `news-100260922111804786`, LLM tách bà Nhi thành một vụ riêng thay vì đặt chung vụ với Hoàng Nato. (2) Thiết kế ontology và `build_own_graph`: `Case` chỉ được gộp khi bài mới có chung bị can với vụ đã biết **tại thời điểm ghi**. Lúc vụ của bà Nhi được tạo, bà chưa xuất hiện trong vụ nào nên được cấp `Case.id` mới; về sau không có bước gộp ngược. So với ontology gợi ý (4 `Case` cho chuyên án này) thì đã giảm còn 2, nhưng chưa hết.
- **Đề xuất sửa:** Trong `build_own_graph` (`src/graph.py`), trích xuất hết 20 bài trước, dùng union-find gộp mọi vụ có chung bị can, rồi mới ghi vào Neo4j. Không tốn thêm token. Đánh đổi: gộp bắc cầu dễ gộp nhầm hơn (A chung người với B, B chung người với C, thì A và C thành một vụ dù không liên quan), và phải giữ toàn bộ kết quả trích xuất trong bộ nhớ.

### Lỗi E4: Phép đo sai — `recall` và `judge` không phạt câu trả lời thừa, và mâu thuẫn nhau

- **Hiện tượng:** Ở Q6 có hai điểm bất thường. (a) Flat được `recall=0.00` nhưng `judge=1`. (b) Graph được điểm tuyệt đối `recall=1.00 judge=2` dù câu trả lời liệt kê 4 vụ trong khi đáp án chuẩn chỉ có 3, và gán thêm một tội danh không đúng cho vụ thứ tư.
- **Bằng chứng:** Trích `ket_qua_benchmark_kg.txt`, Q6, pipeline flat:

```
--- Q6 [aggregation] flat recall=0.00 judge=1 1.65s
- **Vụ việc [2]:** Công an bắt quả tang Thành khi đang mang 5 viên nén màu trắng là ma túy MDMA đi bán.
```

  Q6, pipeline graph:

```
--- Q6 [aggregation] graph recall=1.00 judge=2 3.33s
4.  **Vụ việc Dương Minh Tuấn (Hoàng Nato):**
    *   Dữ kiện knowledge graph xác định vụ việc có liên quan đến chất MDMA.
    *   Tội danh liên quan: Tổ chức sử dụng trái phép chất ma túy (**Điều 255 BLHS**) và Mua bán trái phép chất ma túy (**Điều 251 BLHS**).
```

  `must_include` của Q6 trong `data/benchmark_kg.json` là `["Cái Quang Huy", "Lê Minh Thành", "Pháp y tâm thần"]`. Truy vấn trả lời thẳng câu hỏi trên graph:

```cypher
MATCH (k:Case)-[i:INVOLVES]->(:Substance {name:'MDMA'})
RETURN k.name AS case, i.amount_text AS amount, i.grams AS grams;
```

```
case                                                                amount      grams
Vụ án Viện Pháp y tâm thần Trung ương                               0,686g      0.686
Dương Minh Tuấn (Hoàng Nato)                                                    null
Vụ án Lê Minh Thành và đồng phạm (Hà Nội)                           5 viên      null
Vụ vận chuyển ma túy từ Đức của Cái Quang Huy và Nguyễn Tiến Đạt    hơn 9,6kg   9600.0
```

- **Nguyên nhân:** Nằm ở phép đo, không ở pipeline.
  - (a) `keyword_recall` so chuỗi nguyên văn. Flat mô tả đúng hai vụ (lô MDMA gửi từ Đức, 5 viên MDMA của "Thành") nhưng không viết họ tên đầy đủ "Lê Minh Thành" hay "Cái Quang Huy", nên được 0 dù LLM judge cho là đúng một phần. Ở đây judge hợp lý hơn recall.
  - (b) Cả hai thước đo chỉ kiểm tra **có đủ** ý trong đáp án, không kiểm tra **thừa**. Vụ thứ tư không hẳn sai: bài `news-100260920221957595` viết các đường dây "mua bán ma túy loại etomidate, ketamine, thuốc lắc", và bảng tên chuẩn nối "thuốc lắc" vào MDMA; tức là đáp án chuẩn có thể đang thiếu. Nhưng phần "Mua bán … Điều 251" gán cho "Vụ việc Dương Minh Tuấn" thì lệch với graph: `Charge` của ông chỉ nối tới "tổ chức sử dụng trái phép chất ma túy"; tội mua bán thuộc về người khác trong cùng `Case`. LLM đã trộn tội danh của cả vụ vào một người (đây đồng thời là lỗi nhóm E5), và điểm số không phản ánh điều đó.
- **Đề xuất sửa:** Trong `data/benchmark_kg.json` thêm trường `must_not_include` hoặc số vụ kỳ vọng, và trong `JUDGE_PROMPT` yêu cầu trừ điểm khi có ý không có trong đáp án. Vì không được sửa `bench_kg.py`, trong phạm vi bài này chỉ có thể đọc tay từng câu trả lời. Đánh đổi: thước đo chặt hơn đòi hỏi đáp án chuẩn đầy đủ hơn (phải quyết định "thuốc lắc" có tính là MDMA không), và judge dài hơn thì tốn thêm token chấm.

## 4. Kết luận (5 điểm)

> **Flat RAG là đủ** khi câu trả lời nằm trong một đoạn văn của một nguồn: ở Q1 và Q2 hai pipeline cùng đạt recall 1.00 / judge 2, trong khi Flat chỉ dùng 696 token vào mỗi câu so với 2236 của Graph (×3.21) và không tốn 20 lần gọi LLM (39199 token vào, 7188 token ra, thêm khoảng 80 giây) để dựng graph.
>
> **Nên dùng KG** khi câu hỏi phải ghép nhiều nguồn hoặc gom nhiều tài liệu: ở Q3–Q6 Flat chỉ đạt recall 0.00–0.40 và judge 1, còn Graph đạt 1.00 / 2 ở cả bốn câu; tính trung bình 6 câu là 0.51 / 1.33 so với 1.00 / 2.00. Điều kiện để KG đáng tiền trong bài này: (1) có một thực thể chung ổn định giữa các nguồn (tội danh) và một phía có cấu trúc đủ đều để trích bằng regex (luật), nên chi phí LLM chỉ rơi vào 20 bài báo; (2) câu hỏi thuộc loại xuyên nguồn hoặc tổng hợp chiếm phần đáng kể; (3) số câu hỏi đủ nhiều để chi phí dựng một lần được chia nhỏ.
>
> Hai lưu ý từ số liệu: thiết kế ontology và `context()` quyết định cả độ chính xác lẫn chi phí (ontology gợi ý: recall 0.89, 5621 token vào mỗi câu; ontology tự thiết kế: recall 1.00, 2236 token), và điểm tuyệt đối ở đây chỉ dựa trên 6 câu hỏi với thước đo không phạt câu trả lời thừa (lỗi E4), nên chưa đủ để khẳng định graph không còn lỗi (E3).

## 5. Tự kiểm (5 điểm)

```
$ pytest tests/ -q
................................................                         [100%]
48 passed in 0.07s

$ python bench_kg.py --check
[OK] Dữ liệu: 18 điều luật, 20 bài báo
[OK] KG-1 link_entity
[OK] Neo4j kết nối được
[provider] chat = gemini:gemini-3.1-flash-lite | embedding = gemini:gemini-embedding-001
[OK] KG-2 build_graph: 165 node / 424 cạnh, đường xuyên 2 KB dài 2 cạnh
[OK] KG-3 context: 14 dữ kiện, có Điều 251
[OK] KG-4 GraphRAGAgent.answer
[OK] Chi phí check: 1 lần gọi LLM, $0.00000. Graph nhỏ (luật + 1 bài) vẫn còn trong Neo4j để bạn xem; chạy --judge để dựng graph đầy đủ.
```

Ảnh Neo4j: `report/img/kg_count.png`, `report/img/kg_cross_kb.png`, `report/img/kg_my_case.png`.
Người đã chọn cho `kg_my_case.png`: Trần Thanh Tuấn

## Vấn đề gặp phải (không tính điểm)

> - `gemini-2.5-flash-lite` (model mặc định trong `src/llm.py`) trả 404 "no longer available to new users". Đã chạy bằng biến môi trường `GEMINI_CHAT_MODEL=gemini-3.1-flash-lite`; cả hai file kết quả dùng cùng model này.
> - Key Gemini miễn phí bị giới hạn 100 request embedding mỗi phút, làm `bench_kg.py --judge` dừng với lỗi 429 giữa chừng. Đã thêm `max_retries=12` vào `_openai_client` trong `src/llm.py` để SDK tự chờ và thử lại. Hệ quả: cột `seconds` có lẫn thời gian chờ.
> - Cột USD bằng 0 vì bảng giá không có model đã dùng (xem ghi chú ở mục 1).
> - Dùng Python 3.13 thay vì 3.11 như hướng dẫn; test và benchmark chạy bình thường.
