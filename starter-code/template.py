"""
Lab #4: System Prompt Engineering & Tool Calling Engine

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas + ReAct Loop.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1 (DONE): SYSTEM PROMPT cấp sản xuất
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant: trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (xe điện VinFast, du lịch Vinpearl) kiêm tiếp nhận yêu cầu hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, ngắn gọn, chính xác tuyệt đối về số liệu. Không bịa thông tin, không phỏng đoán, không đưa ra cam kết về giá hay thời gian giao hàng.
- Ngôn ngữ: Luôn trả lời bằng tiếng Việt, xưng "VinAssistant", gọi khách là "Quý khách".

## 2. AVAILABLE TOOLS
{tools}

Tóm tắt:
- `search_product_catalog(category, max_price)` — tra cứu sản phẩm theo danh mục
  ('xe_dien' | 'du_lich') và giá tối đa (VNĐ).
- `submit_support_ticket(customer_name, issue_description, priority)` — tạo ticket hỗ trợ,
  trả về mã ticket dạng TK-YYYYMMDD-NNN.

## 3. CORE RULES (bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (tên, giá, tính năng, tình trạng kho).
   Mọi thông tin sản phẩm PHẢI đến từ observation của `search_product_catalog`.
2. KHÔNG BAO GIỜ tự sinh ticket_id. Mã ticket PHẢI lấy từ `submit_support_ticket`.
3. Nếu tool trả về danh sách rỗng → nói rõ "không tìm thấy sản phẩm phù hợp" và
   gợi ý phương án thay thế. TUYỆT ĐỐI không đề xuất sản phẩm ngoài kết quả tool.
4. Nếu một câu hỏi cần NHIỀU tool (vừa tra cứu vừa tạo ticket) → gọi lần lượt đủ cả hai
   trước khi tổng hợp câu trả lời cuối.
5. Câu hỏi FAQ chính sách chung (bảo hành, sạc pin, đổi trả...) → trả lời trực tiếp,
   không gọi tool.
6. Nếu thiếu tham số bắt buộc (ví dụ khách chưa cho tên) → hỏi lại, không tự điền.
7. Luôn hiển thị giá theo định dạng VNĐ dễ đọc (ví dụ: 548.000.000 VNĐ).

## 4. OPERATIONAL BOUNDARIES
- Chỉ tư vấn trong phạm vi hệ sinh thái Vingroup (VinFast, Vinpearl, VinWonders...).
- Từ chối lịch sự các câu hỏi ngoài phạm vi: chính trị, y tế, pháp lý, đối thủ cạnh tranh.
- Không đưa ra cam kết về giá khuyến mãi, thời gian giao xe hay bồi thường.
- Không thu thập thông tin nhạy cảm (CCCD, số thẻ ngân hàng, mật khẩu).

## 5. OUTPUT CONTRACT
Mỗi bước suy luận tuân thủ đúng định dạng ReAct:

Thought: <suy nghĩ ngắn gọn về việc cần làm tiếp>
Action: <tên tool>({"arg": "value"})
Observation: <kết quả tool trả về>
... (lặp lại tối đa 5 vòng)
Final Answer: <câu trả lời cuối cùng bằng tiếng Việt, chỉ dựa trên Observation>
""".replace("{tools}", json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# FAQ KNOWLEDGE BASE — dùng cho câu hỏi chính sách (không cần tool)
# ═══════════════════════════════════════════════════════════════════════════

FAQ_KNOWLEDGE_BASE: List[Tuple[List[str], str]] = [
    (
        ["bảo hành pin", "pin", "bảo hành"],
        "Chính sách bảo hành pin xe điện VinFast là 10 năm (không giới hạn số km) cho khách hàng "
        "cá nhân. VinFast cam kết thay thế pin miễn phí nếu dung lượng pin (SOH) giảm xuống dưới "
        "70% trong thời gian bảo hành. Xe được bảo hành chính hãng 7-10 năm tùy dòng.",
    ),
    (
        ["sạc", "trạm sạc"],
        "VinFast có hệ thống trạm sạc phủ rộng toàn quốc tại các trung tâm thương mại, chung cư và "
        "trạm xăng đối tác. Quý khách có thể sạc tại nhà bằng bộ sạc di động hoặc sạc nhanh DC tại "
        "trạm công cộng.",
    ),
    (
        ["đổi trả", "hoàn tiền", "hủy phòng", "hủy đặt"],
        "Chính sách đổi trả và hoàn hủy được áp dụng theo điều khoản tại thời điểm đặt. "
        "VinAssistant có thể tạo ticket hỗ trợ để bộ phận CSKH xử lý cụ thể cho trường hợp của Quý khách.",
    ),
]

FAQ_FALLBACK = (
    "VinAssistant chưa có đủ dữ liệu chính thức để trả lời chính xác câu hỏi này. "
    "Quý khách vui lòng cung cấp thêm thông tin, hoặc VinAssistant có thể tạo ticket hỗ trợ "
    "để bộ phận CSKH Vingroup liên hệ trực tiếp."
)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.

    Mục đích: so sánh chất lượng trả lời khi LLM phải "bịa" thông tin (hallucination)
    vì không được truy cập dữ liệu thật.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM 1 lượt duy nhất, KHÔNG kèm tool."""
        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    "Bạn là chatbot tư vấn Vingroup. Trả lời ngắn gọn bằng tiếng Việt.\n\n"
                    f"Câu hỏi: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_baseline",
                }
            except Exception as exc:  # noqa: BLE001 — fallback sang mock
                print(f"[ChatbotBaseline] Live API lỗi ({exc}) → dùng mock.")

        return {
            "answer": (
                f"[Chatbot Baseline — không dùng tool] Về yêu cầu \"{user_input}\": "
                "VinFast hiện có nhiều mẫu xe điện với giá tham khảo khoảng 300–700 triệu đồng, "
                "và Vinpearl có các gói nghỉ dưỡng từ khoảng 3 triệu đồng. "
                "(Cảnh báo: số liệu trên do mô hình tự sinh, KHÔNG tra cứu từ catalog thật "
                "→ đây chính là hiện tượng hallucination.)"
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
      - 2 custom tools: search_product_catalog, submit_support_ticket
      - Sequential & parallel tool calling
      - Max-iterations safeguard
      - Full trace logging (Thought / Action / Observation)
    """

    # --- Từ khóa nhận diện intent ---
    KW_XE_DIEN = ["xe điện", "xe dien", "vinfast", "vf ", "vf3", "vf5", "vf8", "vf9", "ô tô điện", "oto điện"]
    KW_DU_LICH = ["du lịch", "du lich", "resort", "vinpearl", "nghỉ dưỡng", "khách sạn", "tour", "vinwonders"]
    KW_BROWSE = ["xem", "giá", "gia dưới", "dưới", "tư vấn", "mua", "báo giá", "bao nhiêu tiền",
                 "có ... không", "tìm", "gợi ý", "danh sách", "combo", "gói"]
    KW_TICKET = ["lỗi", "hỏng", "bị ", "khiếu nại", "phản hồi", "phàn nàn", "hỗ trợ", "ghi nhận",
                 "sự cố", "không hoạt động", "kém", "ẩm mốc", "complain", "trục trặc"]
    KW_FAQ = ["chính sách", "bảo hành", "kéo dài bao lâu", "quy định", "thủ tục", "điều kiện",
              "như thế nào", "có được không", "bao lâu"]

    PRICE_UNITS = {"tỷ": 1_000_000_000, "tỉ": 1_000_000_000,
                   "triệu": 1_000_000, "trieu": 1_000_000,
                   "nghìn": 1_000, "ngàn": 1_000, "k": 1_000}

    def __init__(self, max_iterations: int = 5, api_key: Optional[str] = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    # ───────────────────────────────────────────────────────────────────
    # TODO 3 (DONE): Intent Detection — rule-based simulator
    # ───────────────────────────────────────────────────────────────────

    def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Phân tích user_input → xác định tool cần gọi và tham số."""
        text = user_input.lower()

        has_xe = any(k in text for k in self.KW_XE_DIEN)
        has_dl = any(k in text for k in self.KW_DU_LICH)
        has_browse = any(k in text for k in ["xem", "giá", "dưới", "mua", "tư vấn", "tìm",
                                             "gợi ý", "danh sách", "báo giá", "bao nhiêu tiền"])
        is_faq_style = any(k in text for k in self.KW_FAQ)

        # --- Catalog intent: phải có danh mục VÀ tín hiệu tra cứu/giá ---
        needs_catalog = (has_xe or has_dl) and has_browse

        # --- Ticket intent: phải có dấu hiệu sự cố/phản hồi ---
        has_issue_kw = any(k in text for k in self.KW_TICKET)
        customer_name = self._extract_customer_name(user_input)
        needs_ticket = has_issue_kw and (customer_name is not None or "hỗ trợ" in text)

        # --- FAQ: hỏi chính sách chung, không tra cứu giá, không báo lỗi ---
        is_faq = is_faq_style and not needs_catalog and not needs_ticket

        # --- Chuẩn bị arguments ---
        catalog_args: Dict[str, Any] = {}
        if needs_catalog:
            if has_dl and not has_xe:
                category = "du_lich"
            elif has_xe and not has_dl:
                category = "xe_dien"
            else:
                # Cả hai: ưu tiên danh mục xuất hiện trước trong câu
                idx_xe = min((text.find(k) for k in self.KW_XE_DIEN if k in text), default=10**9)
                idx_dl = min((text.find(k) for k in self.KW_DU_LICH if k in text), default=10**9)
                category = "xe_dien" if idx_xe <= idx_dl else "du_lich"
            catalog_args = {"category": category}
            max_price = self._extract_max_price(user_input)
            if max_price is not None:
                catalog_args["max_price"] = max_price

        ticket_args: Dict[str, Any] = {}
        if needs_ticket:
            ticket_args = {
                "customer_name": customer_name or "Khách hàng ẩn danh",
                "issue_description": self._extract_issue(user_input, customer_name),
                "priority": self._extract_priority(text),
            }

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
            "catalog_args": catalog_args,
            "ticket_args": ticket_args,
        }

    # --- Helper: trích giá tối đa -------------------------------------
    def _extract_max_price(self, text: str) -> Optional[int]:
        pattern = r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ|triệu|trieu|nghìn|ngàn)"
        constrained = re.search(
            r"(?:dưới|nhỏ hơn|ít hơn|tối đa|không quá|khoảng|tầm|trong tầm)\s*" + pattern,
            text, flags=re.IGNORECASE,
        )
        match = constrained or re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            number = float(match.group(1).replace(".", "").replace(",", "."))
            unit = match.group(2).lower()
            return int(number * self.PRICE_UNITS.get(unit, 1))

        # Trường hợp viết số đầy đủ: "dưới 600000000"
        raw = re.search(r"(?:dưới|tối đa|không quá)\s*(\d{7,})", text, flags=re.IGNORECASE)
        if raw:
            return int(raw.group(1))
        return None

    # --- Helper: trích tên khách hàng ---------------------------------
    def _extract_customer_name(self, text: str) -> Optional[str]:
        patterns = [
            r"tên (?:của )?tôi là\s+([^\.,;:\n]+)",
            r"tôi tên(?: là)?\s+([^\.,;:\n]+)",
            r"mình tên(?: là)?\s+([^\.,;:\n]+)",
            r"tôi là\s+([^\.,;:\n]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                name = " ".join(m.group(1).strip().split()[:5]).strip()
                if name:
                    return name
        return None

    # --- Helper: trích mô tả vấn đề -----------------------------------
    def _extract_issue(self, text: str, customer_name: Optional[str]) -> str:
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        candidates = [s for s in sentences if any(k in s.lower() for k in self.KW_TICKET)]
        issue = candidates[0] if candidates else (sentences[0] if sentences else text)

        # Bỏ mệnh đề giới thiệu tên & mệnh đề mức độ ưu tiên
        issue = re.sub(r"(tên (?:của )?tôi là|tôi tên(?: là)?|mình tên(?: là)?)\s+[^\.,;:\n]+[,;:]?",
                       "", issue, flags=re.IGNORECASE)
        issue = re.sub(r",?\s*mức độ\s+(cao|trung bình|thấp|nghiêm trọng)", "", issue, flags=re.IGNORECASE)
        issue = re.sub(r"^.*?(?:ghi nhận phản hồi|phản hồi)\s*[:,]\s*", "", issue, flags=re.IGNORECASE)
        if customer_name:
            issue = issue.replace(customer_name, "")
        issue = issue.strip(" ,;:-")

        if issue:
            issue = issue[0].upper() + issue[1:]
        return issue or "Khách hàng cần hỗ trợ."

    # --- Helper: trích mức độ ưu tiên ---------------------------------
    def _extract_priority(self, text: str) -> str:
        if any(k in text for k in ["nghiêm trọng", "gấp", "khẩn", "ngay lập tức", "nguy hiểm", "cháy", "mất an toàn"]):
            return "high"
        if any(k in text for k in ["không gấp", "mức độ thấp", "khi nào cũng được", "không vội"]):
            return "low"
        return "medium"

    # --- Helper: FAQ ---------------------------------------------------
    def _answer_faq(self, user_input: str) -> str:
        text = user_input.lower()
        for keywords, answer in FAQ_KNOWLEDGE_BASE:
            if any(k in text for k in keywords):
                return answer
        return FAQ_FALLBACK

    @staticmethod
    def _format_vnd(value: int) -> str:
        return f"{value:,}".replace(",", ".") + " VNĐ"

    # ───────────────────────────────────────────────────────────────────
    # TODO 4 (DONE): Agent Loop
    # ───────────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy ReAct Agent Loop."""
        self.trace = []

        intents = self.detect_intent(user_input)
        self.trace.append({
            "step": "intent_detection",
            "user_input": user_input,
            "intents": {k: v for k, v in intents.items() if k != "catalog_args" or True},
        })

        catalog_done = not intents["needs_catalog"]
        ticket_done = not intents["needs_ticket"]
        catalog_results: List[Dict[str, Any]] = []
        ticket_result: Dict[str, Any] = {}

        iteration = 1
        while iteration <= self.max_iterations:
            # --- Iteration gọi tool #1: search_product_catalog ---
            if not catalog_done:
                args = intents["catalog_args"]
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Khách hỏi về sản phẩm → phải gọi search_product_catalog để lấy dữ liệu thật.",
                    "action": "search_product_catalog",
                    "action_input": args,
                })
                catalog_results = TOOL_MAP["search_product_catalog"](**args)
                self.trace.append({
                    "iteration": iteration,
                    "observation": catalog_results,
                })
                catalog_done = True

            # --- Iteration gọi tool #2: submit_support_ticket ---
            elif not ticket_done:
                args = intents["ticket_args"]
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Khách báo sự cố → phải gọi submit_support_ticket để tạo mã ticket thật.",
                    "action": "submit_support_ticket",
                    "action_input": args,
                })
                ticket_result = TOOL_MAP["submit_support_ticket"](**args)
                self.trace.append({
                    "iteration": iteration,
                    "observation": ticket_result,
                })
                ticket_done = True

            # --- Không còn tool nào cần gọi → tổng hợp Final Answer ---
            if catalog_done and ticket_done:
                answer = self._compose_final_answer(
                    user_input, intents, catalog_results, ticket_result
                )
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Đã đủ observation → tổng hợp Final Answer.",
                    "final_answer": answer,
                })
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }

            iteration += 1

        # --- Safeguard: vượt quá số bước tối đa ---
        self.trace.append({"step": "guard", "note": "max_iterations_reached"})
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa. Vui lòng thử lại với câu hỏi ngắn gọn hơn.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
        }

    # ───────────────────────────────────────────────────────────────────
    # Tổng hợp Final Answer từ observation trong trace
    # ───────────────────────────────────────────────────────────────────

    def _compose_final_answer(
        self,
        user_input: str,
        intents: Dict[str, Any],
        catalog_results: List[Dict[str, Any]],
        ticket_result: Dict[str, Any],
    ) -> str:
        parts: List[str] = []

        # --- Phần 1: kết quả tra cứu catalog ---
        if intents["needs_catalog"]:
            if catalog_results and "error" in (catalog_results[0] or {}):
                parts.append(
                    "Rất tiếc, hệ thống catalog đang tạm thời không truy cập được. "
                    "VinAssistant không tìm thấy dữ liệu sản phẩm để tư vấn."
                )
            elif not catalog_results:
                budget = intents["catalog_args"].get("max_price")
                budget_txt = f" trong tầm giá {self._format_vnd(budget)}" if budget else ""
                parts.append(
                    f"Rất tiếc, VinAssistant không tìm thấy sản phẩm phù hợp{budget_txt}. "
                    "Quý khách có thể tham khảo mức ngân sách cao hơn hoặc để VinAssistant "
                    "ghi nhận nhu cầu để tư vấn viên liên hệ lại."
                )
            else:
                lines = [f"VinAssistant tìm thấy {len(catalog_results)} sản phẩm phù hợp:"]
                for i, p in enumerate(catalog_results, 1):
                    lines.append(
                        f"{i}. {p['name']} — {self._format_vnd(p['price_vnd'])} "
                        f"({p.get('availability', 'n/a')})\n   {p.get('description', '')}"
                    )
                parts.append("\n".join(lines))

        # --- Phần 2: kết quả tạo ticket ---
        if intents["needs_ticket"]:
            if ticket_result.get("ticket_id") and ticket_result.get("status") == "open":
                parts.append(
                    f"VinAssistant đã ghi nhận yêu cầu hỗ trợ của Quý khách "
                    f"{ticket_result['customer_name']}.\n"
                    f"- Mã ticket: {ticket_result['ticket_id']}\n"
                    f"- Nội dung: {ticket_result['issue_description']}\n"
                    f"- Mức ưu tiên: {ticket_result['priority']}\n"
                    f"- Trạng thái: {ticket_result['status']}\n"
                    "Bộ phận CSKH sẽ liên hệ với Quý khách trong thời gian sớm nhất."
                )
            else:
                parts.append(
                    "Rất tiếc, VinAssistant chưa tạo được ticket hỗ trợ. "
                    "Quý khách vui lòng cung cấp lại họ tên và mô tả vấn đề."
                )

        # --- Phần 3: FAQ / không cần tool ---
        if not parts:
            parts.append(self._answer_faq(user_input))

        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN 
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
