"""
Lab #4 — Tool Layer
Hai custom tool cho VinAssistant + JSON Schema mô tả cho LLM.
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

VALID_PRIORITIES = {"low", "medium", "high"}
VALID_CATEGORIES = {"xe_dien", "du_lich"}


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện (sắp xếp theo giá tăng dần).
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")

    # --- Guard 1: file không tồn tại ---
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    # --- Guard 2: file hỏng / không phải JSON hợp lệ ---
    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return [{"error": f"Cannot read product catalog: {exc}"}]

    if not isinstance(products, list):
        return [{"error": "Product catalog format is invalid."}]

    # --- Guard 3: tham số đầu vào ---
    if not category or not isinstance(category, str):
        return [{"error": "Missing required argument: category."}]

    category_norm = category.strip().lower()

    try:
        max_price_val = int(max_price) if max_price is not None else 999999999999
    except (TypeError, ValueError):
        max_price_val = 999999999999

    # --- Lọc theo category + max_price ---
    results = [
        p for p in products
        if str(p.get("category", "")).lower() == category_norm
        and int(p.get("price_vnd", 0)) <= max_price_val
    ]

    results.sort(key=lambda p: p.get("price_vnd", 0))
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def _classify_issue(issue_description: str) -> str:
    """Phân loại ticket đơn giản bằng keyword (warranty / booking / general)."""
    text = (issue_description or "").lower()
    if any(k in text for k in ["bảo hành", "pin", "lỗi", "hỏng", "sửa chữa", "adas"]):
        return "warranty"
    if any(k in text for k in ["đặt phòng", "booking", "resort", "vinpearl", "khách sạn", "phòng"]):
        return "booking"
    return "general"


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # --- Guard: tham số bắt buộc ---
    if not customer_name or not str(customer_name).strip():
        return {"error": "Missing required argument: customer_name.", "status": "failed"}
    if not issue_description or not str(issue_description).strip():
        return {"error": "Missing required argument: issue_description.", "status": "failed"}

    priority_norm = str(priority).strip().lower()
    if priority_norm not in VALID_PRIORITIES:
        priority_norm = "medium"

    # --- Load ticket hiện có (LUÔN load trước khi ghi để không ghi đè) ---
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing_tickets = loaded
        except (json.JSONDecodeError, OSError):
            existing_tickets = []

    # --- Sinh ticket_id: TK-YYYYMMDD-NNN ---
    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    existing_ids = {t.get("ticket_id") for t in existing_tickets if isinstance(t, dict)}
    while ticket_id in existing_ids:  # tránh trùng ID
        seq += 1
        ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": str(customer_name).strip(),
        "issue_description": str(issue_description).strip(),
        "priority": priority_norm,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": _classify_issue(issue_description),
    }

    existing_tickets.append(new_ticket)

    try:
        with open(tickets_file, "w", encoding="utf-8") as f:
            json.dump(existing_tickets, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        return {"error": f"Cannot save ticket: {exc}", "status": "failed"}

    return {
        "ticket_id": ticket_id,
        "customer_name": new_ticket["customer_name"],
        "issue_description": new_ticket["issue_description"],
        "priority": priority_norm,
        "status": "open",
        "created_at": new_ticket["created_at"],
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast hoặc gói du lịch Vinpearl) "
            "theo danh mục và mức giá tối đa. Dùng tool này bất cứ khi nào khách hỏi về "
            "tên sản phẩm, giá, tính năng hoặc tình trạng hàng."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' cho xe điện VinFast, 'du_lich' cho gói du lịch Vinpearl.",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ (ví dụ: 600000000 cho 600 triệu). Bỏ trống nếu khách không giới hạn giá.",
                    "minimum": 0,
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Ghi nhận yêu cầu hỗ trợ / khiếu nại / phản hồi của khách hàng vào hệ thống ticket "
            "và trả về mã ticket. Dùng tool này khi khách báo lỗi, phàn nàn hoặc yêu cầu hỗ trợ kỹ thuật."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên khách hàng, lấy đúng như khách cung cấp.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn vấn đề khách gặp phải.",
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'high' nếu khẩn cấp/nghiêm trọng, 'medium' mặc định, 'low' nếu không gấp.",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
