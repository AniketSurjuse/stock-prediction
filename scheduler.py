"""
scheduler.py — APScheduler that runs ingestion jobs every 2 hours
during NSE market hours (09:00 – 15:30 IST, Mon–Fri).

Run this in a separate terminal alongside the FastAPI server:
    python scheduler.py
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ingest_prices import run_ingestion as ingest_prices
from ingest_news   import run_ingestion as ingest_news
from db import init_db

IST = ZoneInfo("Asia/Kolkata")


def job_prices():
    now = datetime.now(IST)
    print(f"\n[scheduler] {now:%Y-%m-%d %H:%M} IST — running price ingestion")
    ingest_prices(days=1)


def job_news():
    now = datetime.now(IST)
    print(f"\n[scheduler] {now:%Y-%m-%d %H:%M} IST — running news ingestion")
    ingest_news()


def run():
    init_db()

    scheduler = BlockingScheduler(timezone=IST)

    # Price ingestion: every 2 hours, 09:15 to 15:15, Mon–Fri
    scheduler.add_job(
        job_prices,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9,11,13,15",
            minute="15",
            timezone=IST,
        ),
        id="ingest_prices",
        name="Price ingestion (NSE)",
        misfire_grace_time=300,
    )

    # News ingestion: every 2 hours, 09:00 to 15:00, Mon–Fri
    scheduler.add_job(
        job_news,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9,11,13,15",
            minute="0",
            timezone=IST,
        ),
        id="ingest_news",
        name="News scraper",
        misfire_grace_time=300,
    )

    print("[scheduler] Started — jobs will run every 2h during market hours (IST)")
    print("[scheduler] Scheduled jobs:")
    for job in scheduler.get_jobs():
        next_run = getattr(job, "next_fire_time", None) or getattr(job, "next_run_time", "unscheduled")
        print(f"  • {job.name} — next run: {next_run}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] Stopped.")


if __name__ == "__main__":
    run()