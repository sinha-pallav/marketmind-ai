"""
Data loaders for each source format.

Each loader reads a raw file and returns a list of LangChain Document objects.
A Document has two fields:
  - page_content : str  — the text the LLM will read
  - metadata     : dict — source info used for filtering later (e.g. "source", "type")

Design principle: for large tabular files (100K+ rows) we do NOT create one
Document per row. Instead we aggregate into meaningful summaries so the RAG
system can answer real marketing questions like "which products drive the most
revenue?" rather than returning individual transaction lines.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# CSV Loader: customer_segments.csv
# One Document per segment — already richly described, no aggregation needed.
# ---------------------------------------------------------------------------

def load_customer_segments(path: Path) -> List[Document]:
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Customer Segment: {row['segment_name']}\n"
            f"Segment ID: {row['segment_id']}\n"
            f"Size: {row['size']:,} customers\n"
            f"Average Order Value: INR {row['avg_order_value']}\n"
            f"Purchase Frequency: {row['purchase_frequency_per_year']} times/year\n"
            f"Average 12-Month CLV: INR {row['avg_clv_12m']}\n"
            f"Top Categories: {row['top_categories']}\n"
            f"Preferred Channel: {row['preferred_channel']}\n"
            f"Churn Risk: {row['churn_risk']}\n"
            f"Description: {row['description']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(path),
                "type": "customer_segment",
                "segment_id": row["segment_id"],
                "segment_name": row["segment_name"],
                "churn_risk": row["churn_risk"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# CSV Loader: online_retail.csv
# 500K+ rows of transactions. We aggregate into product-level summaries
# (top 150 products by revenue) so each Document is useful and concise.
# ---------------------------------------------------------------------------

def load_online_retail(path: Path, top_n: int = 150) -> List[Document]:
    df = pd.read_csv(path, dtype={"customer_id": "Int64"})

    # Compute revenue per line item
    df["revenue"] = df["quantity"] * df["price"]

    # Aggregate by product (stockcode)
    product_stats = (
        df.groupby(["stockcode", "description"])
        .agg(
            total_revenue=("revenue", "sum"),
            total_units_sold=("quantity", "sum"),
            num_transactions=("invoice", "nunique"),
            num_customers=("customer_id", "nunique"),
            avg_unit_price=("price", "mean"),
            top_country=("country", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(top_n)
    )

    docs = []
    for _, row in product_stats.iterrows():
        text = (
            f"Product Performance Summary\n"
            f"Product Code: {row['stockcode']}\n"
            f"Product Name: {row['description'].strip()}\n"
            f"Total Revenue: £{row['total_revenue']:,.2f}\n"
            f"Units Sold: {row['total_units_sold']:,}\n"
            f"Number of Transactions: {row['num_transactions']:,}\n"
            f"Unique Customers: {row['num_customers']:,}\n"
            f"Average Unit Price: £{row['avg_unit_price']:.2f}\n"
            f"Top Market: {row['top_country']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(path),
                "type": "product_performance",
                "stockcode": str(row["stockcode"]),
                "product_name": row["description"].strip(),
            }
        ))

    # Also create country-level market summaries
    country_stats = (
        df.groupby("country")
        .agg(
            total_revenue=("revenue", "sum"),
            num_customers=("customer_id", "nunique"),
            num_orders=("invoice", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(20)
    )

    for _, row in country_stats.iterrows():
        text = (
            f"Market Revenue Summary\n"
            f"Country/Market: {row['country']}\n"
            f"Total Revenue: £{row['total_revenue']:,.2f}\n"
            f"Unique Customers: {row['num_customers']:,}\n"
            f"Total Orders: {row['num_orders']:,}\n"
            f"Average Revenue per Customer: £{row['total_revenue'] / row['num_customers']:,.2f}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(path),
                "type": "market_summary",
                "country": row["country"],
            }
        ))

    return docs


# ---------------------------------------------------------------------------
# CSV Loader: bank_marketing.csv
# Campaign response data. We create statistical summaries by job type,
# age group, and contact method — answering "which audience converts best?"
# ---------------------------------------------------------------------------

def load_bank_marketing(path: Path) -> List[Document]:
    df = pd.read_csv(path)

    # Binary encode the target
    df["converted"] = (df["subscribed"] == "yes").astype(int)
    overall_rate = df["converted"].mean() * 100

    docs = []

    # Overall campaign summary
    docs.append(Document(
        page_content=(
            f"Campaign Overview\n"
            f"Total Contacts: {len(df):,}\n"
            f"Overall Conversion Rate: {overall_rate:.1f}%\n"
            f"Subscribers Acquired: {df['converted'].sum():,}\n"
            f"Average Age of Contacts: {df['age'].mean():.1f} years\n"
            f"Contact Methods Used: {', '.join(df['contact'].unique())}\n"
            f"Campaign Months: {', '.join(df['month'].unique())}"
        ),
        metadata={"source": str(path), "type": "campaign_overview"}
    ))

    # Conversion rate by job type
    job_conv = (
        df.groupby("job")["converted"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "converted", "count": "total", "mean": "rate"})
        .sort_values("rate", ascending=False)
        .reset_index()
    )
    job_lines = "\n".join(
        f"  {row['job']}: {row['rate']*100:.1f}% ({row['converted']}/{row['total']})"
        for _, row in job_conv.iterrows()
    )
    docs.append(Document(
        page_content=f"Campaign Conversion Rate by Job Type\n{job_lines}",
        metadata={"source": str(path), "type": "campaign_by_job"}
    ))

    # Conversion rate by contact method
    contact_conv = (
        df.groupby("contact")["converted"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "converted", "count": "total", "mean": "rate"})
        .sort_values("rate", ascending=False)
        .reset_index()
    )
    contact_lines = "\n".join(
        f"  {row['contact']}: {row['rate']*100:.1f}% ({row['converted']}/{row['total']})"
        for _, row in contact_conv.iterrows()
    )
    docs.append(Document(
        page_content=f"Campaign Conversion Rate by Contact Method\n{contact_lines}",
        metadata={"source": str(path), "type": "campaign_by_contact"}
    ))

    # Conversion rate by age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    )
    age_conv = (
        df.groupby("age_group", observed=True)["converted"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "converted", "count": "total", "mean": "rate"})
        .sort_values("rate", ascending=False)
        .reset_index()
    )
    age_lines = "\n".join(
        f"  Age {row['age_group']}: {row['rate']*100:.1f}% conversion ({row['total']:,} contacts)"
        for _, row in age_conv.iterrows()
    )
    docs.append(Document(
        page_content=f"Campaign Conversion Rate by Age Group\n{age_lines}",
        metadata={"source": str(path), "type": "campaign_by_age"}
    ))

    return docs


# ---------------------------------------------------------------------------
# JSON Loader: product_catalog.json
# One Document per product — preserves rich description and campaign copy.
# ---------------------------------------------------------------------------

def load_product_catalog(path: Path) -> List[Document]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for product in data["products"]:
        text = (
            f"Product: {product['name']}\n"
            f"Product ID: {product['product_id']}\n"
            f"Category: {product['category']} > {product['subcategory']}\n"
            f"Price: INR {product['price']}\n"
            f"Gross Margin: {product['margin_pct']}%\n"
            f"Stock Available: {product['stock_units']:,} units\n"
            f"Target Segments: {', '.join(product['target_segment'])}\n"
            f"Tags: {', '.join(product['tags'])}\n"
            f"Description: {product['description']}\n"
            f"Campaign Messaging: {product['campaign_messaging']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(path),
                "type": "product_catalog",
                "product_id": product["product_id"],
                "category": product["category"],
                "margin_pct": product["margin_pct"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# PDF Loader: marketing report
# Extract text page by page using pypdf.
# ---------------------------------------------------------------------------

def load_pdf(path: Path) -> List[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": str(path),
                    "type": "marketing_report",
                    "page": page_num,
                    "filename": path.name,
                }
            ))
    return docs


# ---------------------------------------------------------------------------
# Master loader: loads all sources and returns one combined list
# ---------------------------------------------------------------------------

def load_all(data_dir: Path) -> List[Document]:
    raw = data_dir / "raw"
    all_docs = []

    sources = [
        ("Customer segments",    lambda: load_customer_segments(raw / "customers" / "customer_segments.csv")),
        ("Online retail",        lambda: load_online_retail(raw / "transactions" / "online_retail.csv")),
        ("Bank marketing",       lambda: load_bank_marketing(raw / "campaigns" / "bank_marketing.csv")),
        ("Product catalog",      lambda: load_product_catalog(raw / "products" / "product_catalog.json")),
        ("Marketing report PDF", lambda: load_pdf(raw / "reports" / "q1_marketing_report.pdf")),
    ]

    for name, loader_fn in sources:
        print(f"  Loading {name}...", end=" ", flush=True)
        docs = loader_fn()
        print(f"{len(docs)} documents")
        all_docs.extend(docs)

    return all_docs
