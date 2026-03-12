"""
Data preparation script for MarketMind AI.

Downloads real open-source datasets and generates synthetic complementary data
to simulate a full marketing intelligence knowledge base.

Run from the project root with the venv active:
    python data/prepare_data.py

Produces:
    data/raw/transactions/online_retail.csv   - Real UCI e-commerce transactions
    data/raw/campaigns/bank_marketing.csv     - Real UCI campaign response data
    data/raw/customers/customer_segments.csv  - Synthetic customer segments
    data/raw/products/product_catalog.json    - Synthetic product catalog
    data/raw/reports/q1_marketing_report.pdf  - Synthetic strategy report
"""

import json
import random
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
from fpdf import FPDF

DATA_DIR = Path(__file__).parent / "raw"
random.seed(42)


# ---------------------------------------------------------------------------
# 1. Real dataset: UCI Online Retail II
# ---------------------------------------------------------------------------

def download_online_retail():
    out_path = DATA_DIR / "transactions" / "online_retail.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    print("  Downloading UCI Online Retail II dataset (~45 MB)...")
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    response = httpx.get(url, timeout=120, follow_redirects=True)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        # The zip contains an xlsx file — read it with pandas
        xlsx_name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        print(f"  Extracting {xlsx_name}...")
        with z.open(xlsx_name) as f:
            # Read only Year 2009-2010 sheet to keep size manageable
            df = pd.read_excel(f, sheet_name=0, engine="openpyxl")

    # Clean up column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Basic cleaning: drop rows with no customer ID or negative quantity
    df = df.dropna(subset=["customer_id"])
    df = df[df["quantity"] > 0]
    df["customer_id"] = df["customer_id"].astype(int)

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Real dataset: UCI Bank Marketing (campaign response data)
# ---------------------------------------------------------------------------

def download_bank_marketing():
    out_path = DATA_DIR / "campaigns" / "bank_marketing.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    print("  Downloading UCI Bank Marketing dataset...")
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as outer:
        # Outer zip contains bank-additional.zip as a nested archive
        with outer.open("bank-additional.zip") as inner_bytes:
            with zipfile.ZipFile(BytesIO(inner_bytes.read())) as inner:
                with inner.open("bank-additional/bank-additional-full.csv") as f:
                    df = pd.read_csv(f, sep=";")

    # Rename target column to be self-explanatory
    df = df.rename(columns={"y": "subscribed"})

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows -> {out_path}")


# ---------------------------------------------------------------------------
# 3. Synthetic: Customer Segments
# ---------------------------------------------------------------------------

SEGMENTS = [
    {
        "segment_id": "SEG001",
        "segment_name": "High-Value Loyalists",
        "size": 12400,
        "avg_order_value": 285.50,
        "purchase_frequency_per_year": 8.2,
        "avg_clv_12m": 2341.10,
        "top_categories": "Electronics, Premium Apparel, Home Decor",
        "preferred_channel": "Email",
        "churn_risk": "Low",
        "description": (
            "Customers with 3+ years tenure, consistently high spend, and strong "
            "brand affinity. Respond well to exclusive early-access offers and "
            "loyalty rewards. Highly engaged across email and app channels."
        ),
    },
    {
        "segment_id": "SEG002",
        "segment_name": "Rising Stars",
        "size": 28700,
        "avg_order_value": 142.30,
        "purchase_frequency_per_year": 4.5,
        "avg_clv_12m": 640.35,
        "top_categories": "Apparel, Beauty, Fitness",
        "preferred_channel": "Social Media",
        "churn_risk": "Medium",
        "description": (
            "Young, digitally-native customers (25-35) with growing purchase "
            "frequency. Heavily influenced by social proof and user-generated "
            "content. Price-sensitive but willing to pay for trending products. "
            "Primary acquisition channel: Instagram and TikTok ads."
        ),
    },
    {
        "segment_id": "SEG003",
        "segment_name": "Bargain Hunters",
        "size": 45200,
        "avg_order_value": 68.90,
        "purchase_frequency_per_year": 6.1,
        "avg_clv_12m": 420.29,
        "top_categories": "Household, Grocery, Budget Apparel",
        "preferred_channel": "SMS",
        "churn_risk": "High",
        "description": (
            "Price-driven shoppers who primarily purchase during sales events. "
            "High volume but low margin. Respond strongly to discount codes, "
            "flash sales, and bundle offers. Low brand loyalty — easily swayed "
            "by competitor promotions."
        ),
    },
    {
        "segment_id": "SEG004",
        "segment_name": "Dormant Customers",
        "size": 31100,
        "avg_order_value": 95.00,
        "purchase_frequency_per_year": 0.8,
        "avg_clv_12m": 76.00,
        "top_categories": "Mixed",
        "preferred_channel": "Email",
        "churn_risk": "Critical",
        "description": (
            "Previously active customers who have not purchased in 6-18 months. "
            "Win-back campaigns with personalised 'we miss you' messaging and "
            "targeted discounts show 12-15% reactivation rate. Require distinct "
            "re-engagement strategy separate from active customer communications."
        ),
    },
    {
        "segment_id": "SEG005",
        "segment_name": "Corporate Buyers",
        "size": 3800,
        "avg_order_value": 1240.00,
        "purchase_frequency_per_year": 11.3,
        "avg_clv_12m": 14012.00,
        "top_categories": "Office Supplies, Electronics, Bulk Orders",
        "preferred_channel": "Account Manager",
        "churn_risk": "Low",
        "description": (
            "B2B segment purchasing for business use. Require invoice-based "
            "billing, bulk pricing, and dedicated account management. Long sales "
            "cycles but very high lifetime value. Driven by procurement policies, "
            "not promotional campaigns."
        ),
    },
]


def generate_customer_segments():
    out_path = DATA_DIR / "customers" / "customer_segments.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    df = pd.DataFrame(SEGMENTS)
    df.to_csv(out_path, index=False)
    print(f"  Generated {len(df)} customer segments -> {out_path}")


# ---------------------------------------------------------------------------
# 4. Synthetic: Product Catalog (JSON)
# ---------------------------------------------------------------------------

PRODUCTS = [
    {
        "product_id": "PRD001",
        "name": "ProSport Running Shoes X9",
        "category": "Fitness",
        "subcategory": "Footwear",
        "price": 129.99,
        "cost": 42.00,
        "margin_pct": 67.7,
        "stock_units": 4200,
        "target_segment": ["SEG001", "SEG002"],
        "description": (
            "High-performance running shoe with carbon-fibre midsole and adaptive "
            "cushioning. Available in 8 colorways. Best-seller for the 25-40 "
            "active lifestyle segment. Pairs well with ProSport Compression Socks."
        ),
        "campaign_messaging": (
            "Run further, recover faster. Engineered for athletes who refuse to "
            "compromise. Limited edition colourways available this season."
        ),
        "tags": ["running", "performance", "bestseller", "new-arrival"],
    },
    {
        "product_id": "PRD002",
        "name": "HomeBrew Espresso Machine Elite",
        "category": "Home Decor",
        "subcategory": "Kitchen Appliances",
        "price": 349.00,
        "cost": 138.00,
        "margin_pct": 60.5,
        "stock_units": 870,
        "target_segment": ["SEG001", "SEG005"],
        "description": (
            "15-bar pressure espresso machine with built-in grinder and milk "
            "frother. Brushed stainless steel finish. Award-winning design. "
            "Compatible with all coffee pod standards and fresh ground beans."
        ),
        "campaign_messaging": (
            "Café-quality espresso. Every morning. In your kitchen. The HomeBrew "
            "Elite turns your counter into a premium coffee bar."
        ),
        "tags": ["kitchen", "premium", "gift-worthy", "high-margin"],
    },
    {
        "product_id": "PRD003",
        "name": "EcoWear Organic Cotton Tee",
        "category": "Apparel",
        "subcategory": "Tops",
        "price": 34.99,
        "cost": 9.50,
        "margin_pct": 72.8,
        "stock_units": 18500,
        "target_segment": ["SEG002", "SEG003"],
        "description": (
            "100% GOTS-certified organic cotton. Available in 22 colours, sizes "
            "XS-3XL. Carbon-neutral supply chain. Extremely popular with "
            "eco-conscious millennials and Gen Z buyers."
        ),
        "campaign_messaging": (
            "Wear your values. EcoWear basics are good for you and the planet. "
            "Certified organic. Always ethical."
        ),
        "tags": ["sustainable", "basics", "high-volume", "social-friendly"],
    },
    {
        "product_id": "PRD004",
        "name": "SmartDesk Pro Standing Desk",
        "category": "Office Supplies",
        "subcategory": "Furniture",
        "price": 599.00,
        "cost": 210.00,
        "margin_pct": 65.0,
        "stock_units": 320,
        "target_segment": ["SEG001", "SEG005"],
        "description": (
            "Electric height-adjustable desk with memory presets, cable management "
            "tray, and anti-collision sensors. Max load 120kg. Assembly in under "
            "20 minutes. Popular for home office and corporate bulk orders."
        ),
        "campaign_messaging": (
            "Your best work starts with the right setup. SmartDesk Pro adapts to "
            "you — whether you're sitting, standing, or somewhere in between."
        ),
        "tags": ["work-from-home", "premium", "corporate", "health"],
    },
    {
        "product_id": "PRD005",
        "name": "GlowUp Vitamin C Serum",
        "category": "Beauty",
        "subcategory": "Skincare",
        "price": 49.99,
        "cost": 12.00,
        "margin_pct": 76.0,
        "stock_units": 9600,
        "target_segment": ["SEG002"],
        "description": (
            "15% stabilised Vitamin C serum with hyaluronic acid and niacinamide. "
            "Dermatologist-tested. Suitable for all skin types. 30ml airless pump "
            "bottle. Strong repeat purchase rate — 68% of buyers repurchase within "
            "90 days."
        ),
        "campaign_messaging": (
            "Your glow-up starts here. Clinical-grade Vitamin C, finally "
            "affordable. See results in 28 days or your money back."
        ),
        "tags": ["beauty", "skincare", "high-repeat", "ugc-friendly"],
    },
]


def generate_product_catalog():
    out_path = DATA_DIR / "products" / "product_catalog.json"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"products": PRODUCTS, "total": len(PRODUCTS)}, f, indent=2)
    print(f"  Generated {len(PRODUCTS)} products -> {out_path}")


# ---------------------------------------------------------------------------
# 5. Synthetic: Q1 Marketing Strategy Report (PDF)
# ---------------------------------------------------------------------------

PDF_CONTENT = {
    "title": "Q1 2026 Marketing Strategy Report",
    "subtitle": "MarketMind Retail — Campaign Planning & Performance Review",
    "sections": [
        {
            "heading": "Executive Summary",
            "body": (
                "Q1 2026 marks a strategic shift toward personalised, data-driven "
                "campaign execution. Total marketing spend reached INR 4.2 crore "
                "across digital, email, and in-store channels, generating an "
                "overall ROAS of 3.8x. Email campaigns delivered the highest ROI "
                "at 42x, driven by segmented journeys for High-Value Loyalists and "
                "Rising Stars. Social media acquisition costs rose 18% YoY due to "
                "platform CPM inflation, prompting a partial reallocation to "
                "influencer partnerships."
            ),
        },
        {
            "heading": "Campaign Performance by Channel",
            "body": (
                "Email Marketing: 4.2M emails sent across 6 campaigns. Average "
                "open rate 28.4% (industry benchmark: 21%). Click-through rate "
                "3.9%. Revenue attributed: INR 1.8 crore. Best performing campaign: "
                "'Loyalty Rewards Unlock' targeting SEG001 with 41% open rate.\n\n"
                "Social Media (Paid): Total spend INR 1.1 crore. Instagram drove "
                "62% of social conversions. TikTok emerging as cost-efficient "
                "channel for SEG002 with CPA 31% below Instagram. Facebook showing "
                "declining returns for 18-35 demographic.\n\n"
                "SMS Campaigns: 280K messages sent to Bargain Hunter segment "
                "during flash sale events. Conversion rate 6.2%. Revenue: "
                "INR 38 lakhs. SMS remains most effective channel for time-sensitive "
                "promotions."
            ),
        },
        {
            "heading": "Segment Strategy Review",
            "body": (
                "High-Value Loyalists (SEG001): Responded positively to exclusive "
                "product previews sent 48h before public launch. This segment "
                "contributed 34% of total revenue on 8% of customer base. "
                "Recommended action: expand VIP tier with quarterly gifting program.\n\n"
                "Rising Stars (SEG002): Strong response to social proof content "
                "and influencer collaborations. Average order value grew 12% QoQ "
                "when product recommendations were personalised. Cart abandonment "
                "rate at 68% — priority for Q2 retargeting investment.\n\n"
                "Dormant Customers (SEG004): Win-back campaign sent to 15K "
                "dormant customers achieved 13.2% reactivation — above the 10% "
                "target. Key insight: personalised subject lines referencing last "
                "purchase category outperformed generic discount offers by 2.4x."
            ),
        },
        {
            "heading": "Q2 2026 Priorities",
            "body": (
                "1. Launch AI-personalised email journeys using purchase history "
                "and browsing behaviour. Target: lift email revenue 25% QoQ.\n\n"
                "2. Scale TikTok ad spend by 40% for Rising Stars segment. "
                "Test UGC-led creatives against brand-produced content.\n\n"
                "3. Implement cart abandonment automation for SEG002 and SEG003. "
                "Estimated revenue recovery: INR 55 lakhs per quarter.\n\n"
                "4. Develop corporate account marketing programme for SEG005. "
                "Dedicated account manager outreach + quarterly business reviews.\n\n"
                "5. A/B test SMS vs Push notification for flash sale communications "
                "to Bargain Hunters. Reduce SMS spend if push achieves comparable "
                "conversion at lower cost."
            ),
        },
        {
            "heading": "Budget Allocation Recommendation",
            "body": (
                "Proposed Q2 budget: INR 4.8 crore (+14% QoQ).\n\n"
                "Email & CRM: 35% (INR 1.68 crore) — highest ROI channel, "
                "scaling personalisation infrastructure.\n"
                "Paid Social: 28% (INR 1.34 crore) — shifting mix toward TikTok "
                "and creator partnerships.\n"
                "Influencer Marketing: 15% (INR 72 lakhs) — mid-tier influencers "
                "in fitness, beauty, and lifestyle verticals.\n"
                "SMS & Push: 8% (INR 38 lakhs) — optimise for flash sale events.\n"
                "Product Seeding & PR: 7% (INR 34 lakhs).\n"
                "Testing & Experimentation: 7% (INR 34 lakhs)."
            ),
        },
    ],
}


def _clean(text: str) -> str:
    """Replace characters outside latin-1 range for fpdf2 built-in fonts."""
    return (
        text.replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
    )


def generate_marketing_report_pdf():
    out_path = DATA_DIR / "reports" / "q1_marketing_report.pdf"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, _clean(PDF_CONTENT["title"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 8, _clean(PDF_CONTENT["subtitle"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)

    # Metadata line
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Prepared by: Marketing Intelligence Team  |  Date: March 2026  |  Confidential",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # Sections
    for section in PDF_CONTENT["sections"]:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 9, _clean(section["heading"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, _clean(section["body"]))
        pdf.ln(8)

    pdf.output(str(out_path))
    print(f"  Generated PDF report ({len(pdf.pages)} pages) -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== MarketMind AI — Data Preparation ===\n")

    print("[1/5] Online Retail transactions (UCI)...")
    download_online_retail()

    print("[2/5] Bank Marketing campaign data (UCI)...")
    download_bank_marketing()

    print("[3/5] Customer segments (synthetic)...")
    generate_customer_segments()

    print("[4/5] Product catalog (synthetic)...")
    generate_product_catalog()

    print("[5/5] Q1 Marketing Strategy Report PDF (synthetic)...")
    generate_marketing_report_pdf()

    print("\n=== Done! Raw data is in data/raw/ ===\n")
    print("Data layout:")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(DATA_DIR.parent)}  ({size_kb:.0f} KB)")
