"""Knowledge Graph (Neo4j) + GraphRAG over two drug-topic knowledge bases.

Contract (fixed — bench_kg.py and the tests rely on it):
    link_entity(name, known)                       -> one of `known` or None          (TODO KG-1)
    build_graph(graph, law_docs, news_docs, llm_fn)   load both KBs into Neo4j      (TODO KG-2)
        every node created from ONE document carries the property `doc_id`
    Neo4jGraph.context(question, doc_ids)         -> list[str] facts               (TODO KG-3)
    GraphRAGAgent.answer(question, top_k)         -> str                           (TODO KG-4)

Everything else in this file is a HINT: one possible ontology (below). Use it as is, change it,
or design your own — your own ontology + report/ONTOLOGY.md earns the bonus (see SUBMISSION.md).

Suggested ontology (Crime is the bridge between the law KB and the news KB):

    (:Article {id, title, law, doc_id})-[:DEFINES]->(:Crime {name})
    (:Article)-[:HAS_CLAUSE]->(:Clause {id, number, penalty, text})-[:MENTIONS]->(:Substance {name})
    (:Case {name, summary, date, doc_id})-[:CHARGED_WITH]->(:Crime)
    (:Case)-[:INVOLVES {amount}]->(:Substance)
    (:Case)-[:LOCATED_IN]->(:Location {name})
    (:Person {name, aliases})-[:INVOLVED_IN {role, sentence, charge}]->(:Case)

Own ontology (the default; KG_ONTOLOGY=hint switches back to the suggested one, see report/ONTOLOGY.md):

    (:Article {id, title, law, doc_id})-[:DEFINES]->(:Crime {name})
    (:Article)-[:HAS_CLAUSE]->(:Clause {id, number, kind, penalty, header, text, doc_id})
    (:Clause)-[:THRESHOLD {point, min, max, unit, text}]->(:Substance {name, aliases})-[:IS_A]->(:Substance)
    (:Report {doc_id, name, published})-[:REPORTS {summary}]->(:Case {id, name, summary, date, location})
    (:Case)-[:INVOLVES {amount_text, grams}]->(:Substance)
    (:Person {key, name, aliases})-[:FACES]->(:Charge {id, stage, role, sentence, doc_id})-[:IN_CASE]->(:Case)
    (:Charge)-[:OF_CRIME]->(:Crime)
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .models import Document
from .store import EmbeddingStore

# Canonical substance names: the ones BLHS Chương XX lists, plus common ones in Vietnamese news.
SUBSTANCES = ["Heroine", "Cocaine", "Methamphetamine", "Amphetamine", "MDMA", "XLR-11", "Ketamine",
              "cần sa", "thuốc phiện", "côca"]
CLAUSE_START = re.compile(r"^(\d+)\.\s", re.MULTILINE)
FOOTNOTE = re.compile(r"\[\d+\]")

def load_markdown_docs(folder: str | Path) -> list[Document]:
    """Read crawler output (.md with a flat `key: "value"` front matter) into Documents."""
    docs = []
    for path in sorted(Path(folder).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        _, front, body = raw.split("---", 2)
        metadata = {k: json.loads(v) for k, v in re.findall(r'^(\w+): (".*")$', front, re.MULTILINE)}
        docs.append(Document(id=metadata.get("doc_id", path.stem), content=body.strip(), metadata=metadata))
    return docs

def normalize_crime(name: str) -> str:
    """'Tội Mua bán trái phép chất ma túy' -> 'mua bán trái phép chất ma túy'."""
    name = re.sub(r"\s+", " ", name.strip().strip("\"'“”").lower())
    return name.removeprefix("tội ").strip()

def link_entity(name: str, known: list[str], normalize: Callable[[str], str] = normalize_crime,
                cutoff: float = 0.8) -> str | None:
    """Map a free-text mention (e.g. a charge written by a journalist) onto one canonical name in `known`."""
    key = normalize(name or "")
    if not key:
        return None
    by_key: dict[str, str] = {}
    for original in known:
        by_key.setdefault(normalize(original), original)
    if key in by_key:
        return by_key[key]
    close = difflib.get_close_matches(key, list(by_key), n=1, cutoff=cutoff)
    return by_key[close[0]] if close else None

def find_substances(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name in SUBSTANCES if name.lower() in lowered]

# ----------------------------------------------------------------------------------------------
# HINT — suggested ontology: extraction helpers
# ----------------------------------------------------------------------------------------------

def parse_law_article(doc: Document) -> dict[str, Any]:
    """Deterministic (regex) extraction for one 'Điều' — law text is regular enough to skip the LLM."""
    article_id = doc.metadata["article"]                       # "Điều 251 BLHS"
    title = doc.metadata["title"].split(". ", 1)[-1]           # "Tội mua bán trái phép chất ma túy"
    body = FOOTNOTE.sub("", doc.content)
    starts = list(CLAUSE_START.finditer(body))
    clauses = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(body)
        text = body[start.start():end].strip()
        first_line = text.splitlines()[0]
        penalty = re.search(r"\bbị ((?:phạt|tù|cảnh cáo).+?)(?::|$)", first_line)
        clauses.append({
            "id": f"{article_id} khoản {start.group(1)}",
            "number": int(start.group(1)),
            "penalty": penalty.group(1).rstrip(".") if penalty else "",
            "text": text,
            "substances": find_substances(text),
        })
    return {
        "id": article_id,
        "law": doc.metadata.get("law", ""),
        "title": title,
        "doc_id": doc.id,
        "crime": normalize_crime(title) if title.startswith("Tội ") else None,
        "clauses": clauses,
    }

NEWS_EXTRACTION_PROMPT = """Bạn trích xuất knowledge graph từ một bài báo tiếng Việt về ma túy.
Chỉ dùng thông tin có trong bài. Trả về JSON đúng dạng:
{{"cases": [{{
  "name": "tên ngắn của vụ việc, ví dụ: Vụ mua bán 36kg ma túy tại TP.HCM",
  "summary": "1-2 câu tóm tắt",
  "date": "ngày xảy ra/xét xử nếu có, dạng YYYY-MM-DD hoặc chuỗi rỗng",
  "location": "tỉnh/thành phố, chuỗi rỗng nếu không rõ",
  "charges": ["tội danh, BẮT BUỘC chọn đúng nguyên văn từ DANH SÁCH TỘI DANH"],
  "substances": [{{"name": "tên chất, dùng tên chuẩn trong DANH SÁCH CHẤT nếu khớp", "amount": "khối lượng nếu có"}}],
  "people": [{{"name": "họ tên", "aliases": ["biệt danh"], "role": "bị cáo|bị can|nghi phạm|người liên quan|cán bộ",
               "charge": "tội danh của người này (từ DANH SÁCH TỘI DANH) hoặc chuỗi rỗng",
               "sentence": "mức án nếu có, ví dụ: tử hình, 8 năm tù"}}]
}}]}}
Bài không nói về vụ việc cụ thể (tuyên truyền, hội nghị...) thì trả về {{"cases": []}}.

DANH SÁCH TỘI DANH: {crimes}
DANH SÁCH CHẤT: {substances}

Tiêu đề: {title}
Nội dung:
{content}"""

def extract_news_cases(doc: Document, llm_fn: Callable[[str], str], known_crimes: list[str]) -> list[dict]:
    """LLM extraction for one news article; charges are re-linked to law-KB crimes in code."""
    prompt = NEWS_EXTRACTION_PROMPT.format(
        crimes="; ".join(known_crimes), substances=", ".join(SUBSTANCES),
        title=doc.metadata.get("title", ""), content=doc.content[:12000],
    )
    try:
        cases = json.loads(llm_fn(prompt)).get("cases", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    for case in cases:
        case["charges"] = sorted({c for c in (link_entity(x, known_crimes) for x in case.get("charges", [])) if c})
        for person in case.get("people", []):
            person["charge"] = link_entity(person.get("charge") or "", known_crimes) or ""
    return cases

# ----------------------------------------------------------------------------------------------
# Own ontology (report/ONTOLOGY.md): extraction helpers
# ----------------------------------------------------------------------------------------------

OTHER_SOLID, OTHER_LIQUID = "chất ma túy khác ở thể rắn", "chất ma túy khác ở thể lỏng"
UNKNOWN_SUBSTANCE = "ma túy chưa rõ loại"
GENERIC_SUBSTANCES = {"ma túy", "ma tuý", "chất ma túy", "chất ma tuý"}
# Canonical substance -> other names used in the law or in news slang.
SUBSTANCE_ALIASES = {
    "Heroine": ["heroin", "bạch phiến"],
    "Cocaine": ["cocain"],
    "Methamphetamine": ["ma túy đá", "hàng đá", "meth"],
    "Amphetamine": [],
    "MDMA": ["thuốc lắc", "ecstasy", "kẹo", "ma túy kẹo"],
    "XLR-11": [],
    "Ketamine": ["ketamin", "ke", "ma túy ke"],
    "cần sa": ["cỏ", "cỏ Mỹ", "lá cần sa"],
    "nhựa cần sa": [],
    "nhựa thuốc phiện": ["thuốc phiện"],
    "cao côca": [],
    "lá cây côca": [],
    "lá khát": [],
    "quả thuốc phiện khô": [],
    "quả thuốc phiện tươi": [],
    OTHER_SOLID: [],
    OTHER_LIQUID: [],
    "tiền chất ở thể rắn": [],
    "tiền chất ở thể lỏng": [],
}
# Substances BLHS does not name: they fall under a catch-all point ("các chất ma túy khác ở thể rắn").
SUBSTANCE_GROUP = {"Ketamine": OTHER_SOLID}
# Names offered to the extraction LLM: real substances only, not the law's catch-all groups.
NEWS_SUBSTANCES = [name for name in SUBSTANCE_ALIASES if "thể rắn" not in name and "thể lỏng" not in name]
# How a "điểm" names its substances -> canonical names. First match wins.
LAW_SUBSTANCE_GROUPS = [
    ("heroine", ["Heroine", "Cocaine", "Methamphetamine", "Amphetamine", "MDMA", "XLR-11"]),
    ("nhựa thuốc phiện", ["nhựa thuốc phiện", "nhựa cần sa", "cao côca"]),
    ("lá cây côca", ["lá cây côca", "lá khát", "cần sa"]),
    ("quả thuốc phiện khô", ["quả thuốc phiện khô"]),
    ("quả thuốc phiện tươi", ["quả thuốc phiện tươi"]),
    ("chất ma túy khác ở thể rắn", [OTHER_SOLID]),
    ("chất ma túy khác ở thể lỏng", [OTHER_LIQUID]),
    ("tiền chất ở thể rắn", ["tiền chất ở thể rắn"]),
    ("tiền chất ở thể lỏng", ["tiền chất ở thể lỏng"]),
]
POINT = re.compile(r"^([a-zđ])\)\s+(.+)$", re.MULTILINE)
UNIT = r"(gam|kilôgam|mililít)"
RANGE = re.compile(rf"(?:khối lượng|thể tích) t[ừù] ([\d.,]+) {UNIT} đến dưới ([\d.,]+) {UNIT}")
OPEN_RANGE = re.compile(rf"(?:khối lượng|thể tích) ([\d.,]+) {UNIT} trở lên")
AMOUNT = re.compile(r"(\d+(?:[.,]\d+)*)\s*(kilôgam|kilogam|kg|gram|gam|g|tấn|tạ)\b", re.IGNORECASE)
GRAMS_PER = {"kilôgam": 1000, "kilogam": 1000, "kg": 1000, "tấn": 1_000_000, "tạ": 100_000}
STAGES = ["bắt giữ", "khởi tố", "truy tố", "sơ thẩm", "phúc thẩm"]
SUSPECT_ROLES = ["bị cáo", "bị can", "nghi phạm"]

def vn_number(text: str) -> float:
    """'1.200' -> 1200.0, '9,6' -> 9.6, '05' -> 5.0 (Vietnamese decimal comma, dot as thousands separator)."""
    return float(re.sub(r"\.(?=\d{3}(?!\d))", "", text).replace(",", "."))

def parse_grams(amount: str) -> float | None:
    """'hơn 9,6kg' -> 9600.0; None when the amount is not a mass ('5 viên', '2 bánh')."""
    match = AMOUNT.search(amount or "")
    if not match:
        return None
    return round(vn_number(match.group(1)) * GRAMS_PER.get(match.group(2).lower(), 1), 3)

def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip("\"'“”‘’").lower())

def canonical_substance(name: str) -> str:
    """News spelling or slang -> canonical Substance name; unknown names are kept as written."""
    key = normalize_name(name)
    # A news article never states the legal catch-all group; a drug of unknown type stays unknown.
    if key in GENERIC_SUBSTANCES or key.startswith(("chất ma túy khác", "các chất ma túy khác")):
        return UNKNOWN_SUBSTANCE
    by_alias = {alias: canon for canon, aliases in SUBSTANCE_ALIASES.items() for alias in [canon, *aliases]}
    linked = link_entity(name, list(by_alias), normalize=normalize_name, cutoff=0.85)
    return by_alias[linked] if linked else name.strip()

def parse_thresholds(clause_text: str) -> list[dict]:
    """Mass/volume thresholds of one clause, one entry per 'điểm' (regex; grams or millilitres)."""
    thresholds = []
    for point, text in POINT.findall(clause_text):
        closed, opened = RANGE.search(text), OPEN_RANGE.search(text)
        if closed:
            low, unit, high, high_unit = closed.groups()
        elif opened:
            (low, unit), high, high_unit = opened.groups(), None, None
        else:
            continue
        subject = text.lower()
        substances = next((names for marker, names in LAW_SUBSTANCE_GROUPS if marker in subject), [])
        scale = lambda value, u: vn_number(value) * (1000 if u == "kilôgam" else 1)
        thresholds.append({
            "point": point, "substances": substances, "text": text.rstrip(";."),
            "unit": "mililít" if unit == "mililít" else "gam",
            "min": scale(low, unit), "max": scale(high, high_unit) if high else None,
        })
    return thresholds

def own_law_article(article: dict) -> dict:
    """Add to parse_law_article's output: clause kind (penalty tier), header line and thresholds."""
    main = []
    for clause in article["clauses"]:
        header = clause["text"].splitlines()[0]
        clause["header"] = re.sub(r"^\d+\.\s*", "", header).rstrip(":")
        clause["thresholds"] = parse_thresholds(clause["text"])
        if article["crime"] is None:
            clause["kind"] = "quy định"
        elif "còn có thể bị" in header:
            clause["kind"] = "bổ sung"
        elif not clause["penalty"]:
            clause["kind"] = "khác"
        else:
            clause["kind"] = "tăng nặng"
            main.append(clause)
    if main:
        main[-1]["kind"] = "cao nhất"
        main[0]["kind"] = "cơ bản"
    return article

OWN_NEWS_PROMPT = """Bạn trích xuất dữ kiện từ một bài báo tiếng Việt về ma túy để nạp vào knowledge graph.
Chỉ dùng thông tin có trong bài. Trả về JSON đúng dạng:
{{"cases": [{{
  "name": "tên ngắn của vụ việc, có tên nghi phạm chính hoặc địa điểm",
  "summary": "1-2 câu tóm tắt",
  "date": "ngày xảy ra/xét xử nếu có, dạng YYYY-MM-DD hoặc chuỗi rỗng",
  "location": "tỉnh/thành phố, chuỗi rỗng nếu không rõ",
  "substances": [{{"name": "tên chất, dùng tên trong DANH SÁCH CHẤT nếu khớp",
                   "amount": "tổng khối lượng hoặc số lượng chất này trong vụ, giữ số và đơn vị như bài viết (9,6kg; 5 viên), chuỗi rỗng nếu không nêu"}}],
  "people": [{{"name": "họ tên đầy đủ", "aliases": ["biệt danh hoặc tên gọi khác"],
               "role": "bị cáo|bị can|nghi phạm",
               "charges": [{{"crime": "tội danh, chọn đúng nguyên văn từ DANH SÁCH TỘI DANH, chuỗi rỗng nếu bài không nêu",
                            "stage": "bắt giữ|khởi tố|truy tố|sơ thẩm|phúc thẩm",
                            "sentence": "mức án tòa đã tuyên ở giai đoạn này (tử hình, 36 tháng tù), chuỗi rỗng nếu chưa có"}}]}}]
}}]}}
Quy tắc:
- "people" chỉ gồm cá nhân có tên bị bắt, bị khởi tố, bị truy tố hoặc bị xét xử. Không đưa cán bộ, luật sư, nhân chứng,
  nạn nhân, và không gộp một nhóm người thành một phần tử.
- Chất không rõ loại thì ghi "ma túy"; không tự suy ra tên chất.
- Mỗi vụ việc khác nhau là một phần tử riêng. Đoạn cuối bài thường là tin liên quan về một vụ khác: tách thành vụ riêng,
  không gộp người và chất của vụ đó vào vụ chính.
- Một người qua nhiều giai đoạn tố tụng (án sơ thẩm rồi kháng cáo phúc thẩm) thì mỗi giai đoạn là một phần tử trong "charges".
- Bài không nói về vụ việc cụ thể (tuyên truyền, hội nghị...) thì trả về {{"cases": []}}.

DANH SÁCH TỘI DANH: {crimes}
DANH SÁCH CHẤT: {substances}

Tiêu đề: {title}
Nội dung:
{content}"""

def extract_own_cases(doc: Document, llm_fn: Callable[[str], str], known_crimes: list[str]) -> list[dict]:
    """LLM extraction for one news article, then every free-text field is normalized in code."""
    prompt = OWN_NEWS_PROMPT.format(
        crimes="; ".join(known_crimes), substances=", ".join(NEWS_SUBSTANCES),
        title=doc.metadata.get("title", ""), content=doc.content[:12000],
    )
    try:
        raw_cases = json.loads(llm_fn(prompt)).get("cases", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    cases = []
    for raw in raw_cases:
        substances: dict[str, dict] = {}
        for item in raw.get("substances") or []:
            if not (item.get("name") or "").strip():
                continue
            name, amount = canonical_substance(item["name"]), (item.get("amount") or "").strip()
            entry = substances.setdefault(name, {"name": name, "amount": amount, "grams": parse_grams(amount)})
            if entry["grams"] is None and parse_grams(amount) is not None:
                entry.update(amount=amount, grams=parse_grams(amount))
        people = []
        for person in raw.get("people") or []:
            name = (person.get("name") or "").strip()
            if not name or name[0].isdigit():   # "7 công dân Trung Quốc" is a group, not a person
                continue
            # Short one-word aliases ("Thành") are too ambiguous to identify a person across articles.
            aliases = [a.strip() for a in person.get("aliases") or [] if len(a.split()) >= 2 or len(a.strip()) >= 8]
            charges = {}
            for charge in person.get("charges") or [{}]:
                crime = link_entity(charge.get("crime") or "", known_crimes) or ""
                stage = link_entity(charge.get("stage") or "", STAGES, normalize=normalize_name) or "không rõ"
                charges[(crime, stage)] = {"crime": crime, "stage": stage, "sentence": (charge.get("sentence") or "").strip()}
            people.append({
                "name": name, "names": list(dict.fromkeys([name, *aliases])),
                "role": link_entity(person.get("role") or "", SUSPECT_ROLES, normalize=normalize_name) or "nghi phạm",
                "charges": list(charges.values()),
            })
        cases.append({
            "name": (raw.get("name") or "").strip(), "summary": (raw.get("summary") or "").strip(),
            "date": (raw.get("date") or "").strip(), "location": (raw.get("location") or "").strip(),
            "substances": list(substances.values()), "people": people,
        })
    return cases

# ----------------------------------------------------------------------------------------------
# Neo4j
# ----------------------------------------------------------------------------------------------

class Neo4jGraph:
    """Thin wrapper over the official neo4j driver."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def run(self, cypher: str, **params: Any) -> list[dict]:
        records, _, _ = self.driver.execute_query(cypher, params)
        return [record.data() for record in records]

    def reset(self) -> None:
        """Delete every node, relationship and constraint (bench_kg.py calls this before build_graph)."""
        self.run("MATCH (n) DETACH DELETE n")
        for row in self.run("SHOW CONSTRAINTS YIELD name RETURN name"):
            self.run(f"DROP CONSTRAINT `{row['name']}` IF EXISTS")

    def stats(self) -> dict[str, int]:
        nodes = self.run("MATCH (n) RETURN count(n) AS n")[0]["n"]
        rels = self.run("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
        return {"nodes": nodes, "relationships": rels}

    def seed_facts(self, question: str, doc_ids: list[str], skip_labels: tuple[str, ...] = (),
                   limit: int = 60) -> tuple[list[str], list[str]]:
        """Ontology-independent first step: seed nodes + their 1-hop edges as text facts.

        Seeds = nodes whose `doc_id` is in doc_ids, or whose `name`/`aliases` appear in the question.
        Returns (seed elementIds, facts). Nodes with a label in skip_labels are left out of the facts.
        """
        seeds = self.run(
            """
            MATCH (n)
            WHERE n.doc_id IN $doc_ids
               OR (n.name IS :: STRING AND size(n.name) >= 3 AND toLower($q) CONTAINS toLower(n.name))
               OR any(a IN coalesce(n.aliases, []) WHERE size(a) >= 3 AND toLower($q) CONTAINS toLower(a))
            RETURN elementId(n) AS id
            """,
            q=question, doc_ids=doc_ids,
        )
        seed_ids = [row["id"] for row in seeds]
        edges = self.run(
            """
            MATCH (s)-[r]-(m)
            WHERE elementId(s) IN $ids
              AND none(l IN labels(s) + labels(m) WHERE l IN $skip)
            WITH DISTINCT r LIMIT $limit
            WITH startNode(r) AS a, r, endNode(r) AS b
            RETURN labels(a)[0] AS a_label, coalesce(a.name, a.id) AS a_name, type(r) AS rel,
                   properties(r) AS props, labels(b)[0] AS b_label, coalesce(b.name, b.id) AS b_name
            """,
            ids=seed_ids, skip=list(skip_labels), limit=limit,
        )
        facts = []
        for e in edges:
            props = ", ".join(f"{k}: {v}" for k, v in e["props"].items() if v)
            facts.append(f"({e['a_label']}: {e['a_name']}) -[{e['rel']}{' {' + props + '}' if props else ''}]-> "
                         f"({e['b_label']}: {e['b_name']})")
        return seed_ids, facts

    # ---------------------------------------------------------------- HINT — suggested ontology: writes

    def suggested_constraints(self) -> None:
        for label, key in [("Article", "id"), ("Clause", "id"), ("Crime", "name"), ("Case", "name"),
                           ("Substance", "name"), ("Person", "name"), ("Location", "name")]:
            self.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")

    def add_law_article(self, article: dict) -> None:
        self.run(
            """
            MERGE (a:Article {id: $id}) SET a.title = $title, a.law = $law, a.doc_id = $doc_id
            FOREACH (crime IN CASE WHEN $crime IS NULL THEN [] ELSE [$crime] END |
                MERGE (c:Crime {name: crime}) MERGE (a)-[:DEFINES]->(c))
            WITH a
            UNWIND $clauses AS clause
            MERGE (cl:Clause {id: clause.id})
              SET cl.number = clause.number, cl.penalty = clause.penalty, cl.text = clause.text, cl.doc_id = $doc_id
            MERGE (a)-[:HAS_CLAUSE]->(cl)
            FOREACH (s IN clause.substances | MERGE (sub:Substance {name: s}) MERGE (cl)-[:MENTIONS]->(sub))
            """,
            **article,
        )

    def add_news_case(self, case: dict, doc: Document) -> None:
        self.run(
            """
            MERGE (k:Case {name: $name})
              SET k.summary = $summary, k.date = $date, k.doc_id = $doc_id, k.source_title = $title
            FOREACH (loc IN CASE WHEN $location = '' THEN [] ELSE [$location] END |
                MERGE (l:Location {name: loc}) MERGE (k)-[:LOCATED_IN]->(l))
            FOREACH (crime IN $charges | MERGE (c:Crime {name: crime}) MERGE (k)-[:CHARGED_WITH]->(c))
            FOREACH (s IN $substances | MERGE (sub:Substance {name: s.name}) MERGE (k)-[r:INVOLVES]->(sub)
                SET r.amount = s.amount)
            FOREACH (p IN $people | MERGE (person:Person {name: p.name})
                SET person.aliases = coalesce(p.aliases, [])
                MERGE (person)-[r:INVOLVED_IN]->(k) SET r.role = p.role, r.charge = p.charge, r.sentence = p.sentence)
            """,
            name=case.get("name") or doc.metadata.get("title", doc.id),
            summary=case.get("summary", ""), date=case.get("date", ""), location=case.get("location", ""),
            charges=case.get("charges", []), people=[p for p in case.get("people", []) if p.get("name")],
            substances=[s for s in case.get("substances", []) if s.get("name")],
            doc_id=doc.id, title=doc.metadata.get("title", ""),
        )

    # ---------------------------------------------------------------- KG-3

    def context(self, question: str, doc_ids: list[str], max_facts: int = 60) -> list[str]:
        """Graph facts for a question: seeds + 1 hop, then the legal basis of every case reached."""
        if os.getenv("KG_ONTOLOGY", "own") == "hint":
            return self.suggested_context(question, doc_ids, max_facts)
        return self.own_context(question, doc_ids, max_facts)

    def own_context(self, question: str, doc_ids: list[str], max_facts: int = 60) -> list[str]:
        """Own ontology: Person -> Charge -> Crime <- Article -> Clause, clause picked by mass threshold."""
        seed_ids, edge_facts = self.seed_facts(question, doc_ids, skip_labels=("Clause", "Charge"), limit=30)
        person_ids = [r["id"] for r in self.run(
            "MATCH (p:Person) WHERE elementId(p) IN $ids RETURN elementId(p) AS id", ids=seed_ids)]
        # A question that names people is about their cases only; otherwise every case near a seed counts.
        cases = self.run(
            """
            MATCH (n) WHERE elementId(n) IN $ids
            OPTIONAL MATCH (n)-[:REPORTS|IN_CASE|INVOLVES]-(near:Case)
            OPTIONAL MATCH (n)-[:FACES]->(:Charge)-[:IN_CASE]->(own:Case)
            WITH collect(CASE WHEN size($pids) = 0 THEN near END) + collect(own)
                 + collect(CASE WHEN n:Case THEN n END) AS found
            UNWIND found AS k
            WITH DISTINCT k
            RETURN elementId(k) AS id, k.name AS name, k.summary AS summary, k.location AS location
            """,
            ids=seed_ids, pids=person_ids,
        )
        case_ids = [k["id"] for k in cases]
        facts = [f"Vụ việc '{k['name']}'{' tại ' + k['location'] if k['location'] else ''}: {k['summary']}"
                 for k in cases]

        charges = self.run(
            """
            MATCH (p:Person)-[:FACES]->(ch:Charge)-[:IN_CASE]->(k:Case)
            WHERE elementId(k) IN $case_ids
            OPTIONAL MATCH (ch)-[:OF_CRIME]->(c:Crime)
            OPTIONAL MATCH (c)<-[:DEFINES]-(a:Article)
            RETURN k.name AS case, p.name AS person, p.aliases AS aliases, ch.role AS role, ch.stage AS stage,
                   ch.sentence AS sentence, c.name AS crime, a.id AS article, elementId(p) IN $pids AS named
            ORDER BY named DESC, case, person LIMIT 40
            """,
            case_ids=case_ids, pids=person_ids,
        )
        for c in charges:
            parts = [f"Vụ việc '{c['case']}': {c['person']}"
                     + (f" (còn gọi: {', '.join(c['aliases'])})" if c["aliases"] else ""),
                     f"vai trò: {c['role']}", f"giai đoạn tố tụng: {c['stage']}"]
            if c["crime"]:
                parts.append(f"tội danh: {c['crime']}" + (f" ({c['article']})" if c["article"] else ""))
            if c["sentence"]:
                parts.append(f"mức án: {c['sentence']}")
            facts.append(" | ".join(parts))

        involved = self.run(
            """
            MATCH (k:Case)-[i:INVOLVES]->(s:Substance) WHERE elementId(k) IN $case_ids
            RETURN k.name AS case, s.name AS substance, i.amount_text AS amount ORDER BY case, substance
            """,
            case_ids=case_ids,
        )
        facts += [f"Vụ việc '{i['case']}' liên quan chất {i['substance']}"
                  + (f", khối lượng/số lượng: {i['amount']}" if i["amount"] else "") for i in involved]

        # Bridge: the crimes charged in those cases (of the named people, when there are any) -> Articles.
        pairs = self.run(
            """
            MATCH (k:Case)<-[:IN_CASE]-(ch:Charge)-[:OF_CRIME]->(:Crime)<-[:DEFINES]-(a:Article)
            WHERE elementId(k) IN $case_ids
              AND (size($pids) = 0 OR EXISTS { MATCH (p:Person)-[:FACES]->(ch) WHERE elementId(p) IN $pids })
            RETURN DISTINCT elementId(k) AS case_id, elementId(a) AS article_id
            """,
            case_ids=case_ids, pids=person_ids,
        )
        numbers = re.findall(r"[Đđ]iều (\d+)", question)
        frames = self.run(
            """
            MATCH (a:Article)-[:HAS_CLAUSE]->(cl:Clause)
            WHERE (elementId(a) IN $article_ids OR any(n IN $numbers WHERE a.id STARTS WITH 'Điều ' + n + ' '))
              AND cl.kind IN ['cơ bản', 'cao nhất']
            RETURN a.id AS article, a.title AS title, cl.number AS number, cl.kind AS kind, cl.header AS header
            ORDER BY article, number
            """,
            article_ids=list({p["article_id"] for p in pairs}), numbers=numbers,
        )
        facts += [f"[{f['article']} - {f['title']}] khoản {f['number']} (khung {f['kind']}): {f['header']}"
                  for f in frames]

        # The clause that applies to the mass seized in the case: min <= grams < max.
        applied = self.run(
            """
            UNWIND $pairs AS pair
            MATCH (k:Case) WHERE elementId(k) = pair.case_id
            MATCH (a:Article) WHERE elementId(a) = pair.article_id
            MATCH (k)-[i:INVOLVES]->(s:Substance)-[:IS_A*0..1]->(g:Substance)<-[t:THRESHOLD]-(cl:Clause)
                  <-[:HAS_CLAUSE]-(a)
            WHERE i.grams IS NOT NULL AND t.unit = 'gam' AND t.min <= i.grams AND (t.max IS NULL OR i.grams < t.max)
            RETURN DISTINCT k.name AS case, a.id AS article, cl.number AS number, cl.penalty AS penalty,
                   t.point AS point, t.text AS rule, s.name AS substance, g.name AS grp, i.amount_text AS amount
            ORDER BY article, number
            """,
            pairs=pairs,
        )
        for r in applied:
            group = f", xếp vào nhóm '{r['grp']}'" if r["grp"] != r["substance"] else ""
            facts.append(f"[{r['article']}] Vụ việc '{r['case']}' có {r['amount']} {r['substance']}{group} "
                         f"=> áp dụng khoản {r['number']} điểm {r['point']} ({r['rule']}): {r['penalty']}")

        substances = find_substances(question)
        if numbers and substances:
            rules = self.run(
                """
                MATCH (a:Article)-[:HAS_CLAUSE]->(cl:Clause)-[t:THRESHOLD]->(s:Substance)
                WHERE any(n IN $numbers WHERE a.id STARTS WITH 'Điều ' + n + ' ') AND s.name IN $substances
                RETURN DISTINCT a.id AS article, cl.number AS number, t.point AS point, t.text AS rule,
                       cl.penalty AS penalty
                ORDER BY article, number
                """,
                numbers=numbers, substances=substances,
            )
            facts += [f"[{r['article']}] khoản {r['number']} điểm {r['point']}: {r['rule']} => {r['penalty']}"
                      for r in rules]
        return list(dict.fromkeys(facts + edge_facts))[:max_facts]

    # ---------------------------------------------------------------- own ontology: writes

    def own_constraints(self) -> None:
        for label, key in [("Article", "id"), ("Clause", "id"), ("Crime", "name"), ("Substance", "name"),
                           ("Person", "key"), ("Case", "id"), ("Charge", "id"), ("Report", "doc_id")]:
            self.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")

    def add_substances(self) -> None:
        self.run(
            """
            UNWIND $rows AS row
            MERGE (s:Substance {name: row.name}) SET s.aliases = row.aliases
            FOREACH (parent IN row.is_a | MERGE (g:Substance {name: parent}) MERGE (s)-[:IS_A]->(g))
            """,
            rows=[{"name": name, "aliases": aliases, "is_a": [SUBSTANCE_GROUP[name]] if name in SUBSTANCE_GROUP else []}
                  for name, aliases in SUBSTANCE_ALIASES.items()],
        )

    def add_own_article(self, article: dict) -> None:
        self.run(
            """
            MERGE (a:Article {id: $id}) SET a.title = $title, a.law = $law, a.doc_id = $doc_id
            FOREACH (crime IN CASE WHEN $crime IS NULL THEN [] ELSE [$crime] END |
                MERGE (c:Crime {name: crime}) MERGE (a)-[:DEFINES]->(c))
            WITH a
            UNWIND $clauses AS clause
            MERGE (cl:Clause {id: clause.id})
              SET cl.number = clause.number, cl.kind = clause.kind, cl.penalty = clause.penalty,
                  cl.header = clause.header, cl.text = clause.text, cl.doc_id = $doc_id
            MERGE (a)-[:HAS_CLAUSE]->(cl)
            WITH cl, clause
            UNWIND clause.thresholds AS t
            UNWIND t.substances AS name
            MERGE (s:Substance {name: name})
            MERGE (cl)-[r:THRESHOLD {point: t.point}]->(s)
              SET r.min = t.min, r.max = t.max, r.unit = t.unit, r.text = t.text
            """,
            **article,
        )

    def add_own_case(self, case_id: str, case: dict, doc: Document) -> None:
        self.run(
            """
            MERGE (r:Report {doc_id: $doc_id}) SET r.name = $title, r.published = $published
            MERGE (k:Case {id: $case_id})
              ON CREATE SET k.name = $name, k.summary = $summary, k.date = $date, k.location = $location
            MERGE (r)-[rep:REPORTS]->(k) SET rep.summary = $summary
            FOREACH (s IN $substances | MERGE (sub:Substance {name: s.name}) MERGE (k)-[i:INVOLVES]->(sub)
                ON CREATE SET i.amount_text = s.amount, i.grams = s.grams
                FOREACH (_ IN CASE WHEN i.grams IS NULL AND s.grams IS NOT NULL THEN [1] ELSE [] END |
                    SET i.amount_text = s.amount, i.grams = s.grams))
            WITH k
            UNWIND $people AS p
            MERGE (person:Person {key: p.key}) ON CREATE SET person.name = p.name, person.aliases = []
            SET person.aliases = person.aliases
                + [n IN p.names WHERE NOT n IN person.aliases AND toLower(n) <> toLower(person.name)]
            WITH k, person, p
            UNWIND p.charges AS c
            MERGE (ch:Charge {id: c.id})
              ON CREATE SET ch.doc_id = $doc_id, ch.stage = c.stage, ch.role = p.role, ch.sentence = c.sentence
            FOREACH (_ IN CASE WHEN c.sentence <> '' THEN [1] ELSE [] END | SET ch.sentence = c.sentence)
            MERGE (person)-[:FACES]->(ch)
            MERGE (ch)-[:IN_CASE]->(k)
            FOREACH (crime IN CASE WHEN c.crime = '' THEN [] ELSE [c.crime] END |
                MERGE (cr:Crime {name: crime}) MERGE (ch)-[:OF_CRIME]->(cr))
            """,
            case_id=case_id, name=case["name"] or doc.metadata.get("title", doc.id), summary=case["summary"],
            date=case["date"], location=case["location"], substances=case["substances"], people=case["people"],
            doc_id=doc.id, title=doc.metadata.get("title", ""), published=doc.metadata.get("document_version", ""),
        )

    def suggested_context(self, question: str, doc_ids: list[str], max_facts: int = 60) -> list[str]:
        """Retrieval over the suggested ontology: Case -> Crime <- Article -> Clause."""
        seed_ids, facts = self.seed_facts(question, doc_ids)
        cases = self.run(
            """
            MATCH (k:Case)
            WHERE elementId(k) IN $ids OR EXISTS { MATCH (s)--(k) WHERE elementId(s) IN $ids }
            RETURN elementId(k) AS id, k.name AS name, k.summary AS summary
            """,
            ids=seed_ids,
        )
        facts += [f"Vụ việc '{k['name']}': {k['summary']}" for k in cases]
        clauses = self.run(
            """
            MATCH (k:Case)-[:CHARGED_WITH]->(:Crime)<-[:DEFINES]-(a:Article)-[:HAS_CLAUSE]->(cl:Clause)
            WHERE elementId(k) IN $case_ids
              AND (cl.number = 1 OR EXISTS { (k)-[:INVOLVES]->(:Substance)<-[:MENTIONS]-(cl) })
            RETURN DISTINCT a.id AS article, a.title AS title, cl.number AS number, cl.text AS text
            ORDER BY article, number
            """,
            case_ids=[k["id"] for k in cases],
        )
        numbers = re.findall(r"[Đđ]iều (\d+)", question)
        if numbers:
            clauses += self.run(
                """
                MATCH (a:Article)-[:HAS_CLAUSE]->(cl:Clause)
                WHERE any(n IN $numbers WHERE a.id STARTS WITH 'Điều ' + n + ' ')
                  AND (cl.number = 1 OR EXISTS { MATCH (cl)-[:MENTIONS]->(s:Substance) WHERE s.name IN $substances })
                RETURN DISTINCT a.id AS article, a.title AS title, cl.number AS number, cl.text AS text
                ORDER BY article, number
                """,
                numbers=numbers, substances=find_substances(question),
            )
        facts += [f"[{c['article']} - {c['title']}] khoản {c['number']}: {c['text']}" for c in clauses]
        return list(dict.fromkeys(facts))[:max_facts]

# ---------------------------------------------------------------------------------------------- KG-2

def build_graph(graph: Neo4jGraph, law_docs: list[Document], news_docs: list[Document],
                llm_fn: Callable[..., str]) -> None:
    """Load both KBs into an empty graph. llm_fn(prompt, json_mode=False) -> str (metered OpenAI chat)."""
    if os.getenv("KG_ONTOLOGY", "own") == "hint":
        build_suggested_graph(graph, law_docs, news_docs, llm_fn)
    else:
        build_own_graph(graph, law_docs, news_docs, llm_fn)

def build_own_graph(graph: Neo4jGraph, law_docs: list[Document], news_docs: list[Document],
                    llm_fn: Callable[..., str]) -> None:
    """Own ontology (report/ONTOLOGY.md). Cases and people are resolved across articles before writing."""
    graph.own_constraints()
    graph.add_substances()
    articles = [own_law_article(parse_law_article(d)) for d in law_docs]
    for article in articles:
        graph.add_own_article(article)
    crimes = [a["crime"] for a in articles if a["crime"]]
    person_index: dict[str, str] = {}   # any name/alias key -> Person.key
    person_case: dict[str, str] = {}    # Person.key -> Case.id of the first case that person appeared in
    for doc in news_docs:
        cases = extract_own_cases(doc, lambda p: llm_fn(p, json_mode=True), crimes)
        for index, case in enumerate(cases):
            for person in case["people"]:
                keys = [normalize_name(n) for n in person["names"]]
                person["key"] = next((person_index[k] for k in keys if k in person_index), keys[0])
                for key in keys:
                    person_index.setdefault(key, person["key"])
            known = [person_case[p["key"]] for p in case["people"] if p["key"] in person_case]
            case_id = max(set(known), key=known.count) if known else f"{doc.id}#{index}"
            for person in case["people"]:
                # A later article that only mentions a known suspect adds nothing: keep the charges already stored.
                if person_case.get(person["key"]) == case_id:
                    person["charges"] = [c for c in person["charges"] if c["crime"] or c["sentence"]]
                person_case.setdefault(person["key"], case_id)
                for charge in person["charges"]:
                    charge["id"] = "|".join([case_id, person["key"], charge["crime"], charge["stage"]])
            graph.add_own_case(case_id, case, doc)
    # An article that names no crime leaves an empty Charge; drop it once another article supplied the crime.
    graph.run(
        """
        MATCH (p:Person)-[:FACES]->(empty:Charge)-[:IN_CASE]->(k:Case)
        WHERE NOT (empty)-[:OF_CRIME]->() AND coalesce(empty.sentence, '') = ''
          AND EXISTS { MATCH (p)-[:FACES]->(other:Charge)-[:IN_CASE]->(k) WHERE (other)-[:OF_CRIME]->() }
        DETACH DELETE empty
        """
    )

def build_suggested_graph(graph: Neo4jGraph, law_docs: list[Document], news_docs: list[Document],
                          llm_fn: Callable[..., str]) -> None:
    """The suggested ontology, kept as the baseline for ket_qua_benchmark_kg.hint.txt."""
    graph.suggested_constraints()
    articles = [parse_law_article(d) for d in law_docs]
    for article in articles:
        graph.add_law_article(article)
    crimes = [a["crime"] for a in articles if a["crime"]]
    for doc in news_docs:
        for case in extract_news_cases(doc, lambda p: llm_fn(p, json_mode=True), crimes):
            graph.add_news_case(case, doc)

# ---------------------------------------------------------------------------------------------- KG-4

GRAPH_PROMPT = """Trả lời câu hỏi chỉ dựa trên ngữ cảnh (đoạn văn bản và dữ kiện từ knowledge graph).
Nêu rõ số Điều luật khi có. Nếu ngữ cảnh không đủ, nói không đủ thông tin.

Dữ kiện knowledge graph:
{facts}

Đoạn văn bản:
{chunks}

Câu hỏi: {question}
Trả lời:"""

class GraphRAGAgent:
    """Hybrid GraphRAG: the same vector top-k as flat RAG, plus facts expanded from the graph."""

    def __init__(self, store: EmbeddingStore, graph: Neo4jGraph, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.graph = graph
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        chunks = self.store.search(question, top_k=top_k)
        doc_ids = list(dict.fromkeys(chunk["metadata"]["doc_id"] for chunk in chunks))
        facts = self.graph.context(question, doc_ids)
        prompt = GRAPH_PROMPT.format(
            facts="\n".join(f"- {fact}" for fact in facts),
            chunks="\n\n".join(f"[{i}] {chunk['content']}" for i, chunk in enumerate(chunks, start=1)),
            question=question,
        )
        return self.llm_fn(prompt)
