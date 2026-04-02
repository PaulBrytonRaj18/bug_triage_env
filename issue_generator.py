# bug_triage_env/issue_generator.py
#
# Generates synthetic GitHub-style issues for the Bug Triage Environment.
#
# Every issue has a ground truth label and severity baked in.
# These are never shown to the agent — they are only used by the graders.
#
# generate_issue(template, issue_id)  → single issue dict
# generate_inbox(task_difficulty)     → ordered list of issue dicts
#
# Difficulty controls inbox size and complexity:
#   easy   — 8 issues, all clearly one type, no duplicates
#   medium — 15 issues, 2 hidden duplicate pairs
#   hard   — 20 issues, subtle duplicates, ambiguous labels, P0 traps

from __future__ import annotations
import random
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

COMPONENTS = [
    "auth", "payment", "dashboard", "api",
    "database", "notifications", "search", "profile",
]

REPORTERS = [
    "alice_dev", "bob_qa", "charlie_user", "diana_pm",
    "eve_support", "frank_external", "grace_tester", "henry_lead",
]

# Ground-truth issue templates.
# Each entry contains everything needed to generate a realistic issue
# AND the hidden answer key (correct_label, correct_severity).
#
# Fields:
#   title           — issue title as a reporter would write it
#   body            — full description, steps, stack traces
#   correct_label   — the right label (what the grader expects)
#   correct_severity— the right severity
#   has_stack_trace — used in observation so agent has a signal
#   components      — systems mentioned (shown in observation)

ISSUE_TEMPLATES: List[Dict] = [

    # -----------------------------------------------------------------------
    # BUGS — P0 (Critical)
    # -----------------------------------------------------------------------
    {
        "title": "Login fails with 500 error on mobile Safari",
        "body": (
            "Steps to reproduce:\n"
            "1. Open the app on iPhone 13 running iOS 16\n"
            "2. Enter valid credentials\n"
            "3. Tap the Login button\n\n"
            "Expected: User is logged in and redirected to dashboard\n"
            "Actual: 'Internal Server Error' shown, user stays on login page\n\n"
            "This affects ALL mobile Safari users. Desktop Chrome works fine.\n\n"
            "Stack trace from server logs:\n"
            "  Error: Cannot read property 'token' of undefined\n"
            "    at AuthService.validateToken (auth.js:142:18)\n"
            "    at middleware.verifySession (app.js:87:12)\n"
            "    at Layer.handle [as handle_request] (router/layer.js:95:5)"
        ),
        "correct_label": "label_bug",
        "correct_severity": "P0",
        "has_stack_trace": True,
        "components": ["auth"],
    },
    {
        "title": "Payment processing hangs indefinitely for amounts over $999",
        "body": (
            "Reproducible every time:\n"
            "- Any payment amount UNDER $999 processes normally (~2s)\n"
            "- Any payment OVER $999 spins forever with no error message\n"
            "- Backend logs show a 30-second timeout then silence\n"
            "- Users are charged but order is not confirmed\n\n"
            "Started after Tuesday's deploy (commit abc1234).\n\n"
            "Stack trace:\n"
            "  TimeoutError: Gateway did not respond within 30000ms\n"
            "    at PaymentGateway.charge (payment.js:203:11)\n"
            "    at OrderService.processCheckout (orders.js:78:5)"
        ),
        "correct_label": "label_bug",
        "correct_severity": "P0",
        "has_stack_trace": True,
        "components": ["payment"],
    },
    {
        "title": "All users logged out after server restart — sessions not persisted",
        "body": (
            "After last night's maintenance window, all active sessions were invalidated.\n"
            "Users lost unsaved work. Session store appears to be in-memory only.\n\n"
            "Redis connection logs show:\n"
            "  WARN: Redis not configured, falling back to MemoryStore\n"
            "  ERROR: MemoryStore does not persist across process restarts\n\n"
            "This is a data-loss incident. All users (~12,000) were affected."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P0",
        "has_stack_trace": False,
        "components": ["auth", "database"],
    },

    # -----------------------------------------------------------------------
    # BUGS — P1 (High)
    # -----------------------------------------------------------------------
    {
        "title": "Dashboard charts not loading for accounts with more than 1000 records",
        "body": (
            "Users with large datasets see completely empty charts on the main dashboard.\n"
            "Network tab shows /api/analytics returning a 504 Gateway Timeout after 60s.\n\n"
            "Affects all enterprise customers (estimated 200+ accounts).\n"
            "Standard accounts (under 1000 records) load fine.\n\n"
            "Started after last Tuesday's deploy. No stack trace visible client-side."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P1",
        "has_stack_trace": False,
        "components": ["dashboard", "api"],
    },
    {
        "title": "Email notifications sent twice for every payment event",
        "body": (
            "Users are receiving duplicate confirmation emails for every payment.\n"
            "Checked server logs: notification service fires twice when the payment\n"
            "webhook is retried after a temporary network blip.\n\n"
            "Root cause: no idempotency check on the notification trigger.\n"
            "Impact: every user who made a payment in the last 3 days got 2 emails.\n\n"
            "No stack trace. Logic error in notification/payment integration."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P1",
        "has_stack_trace": False,
        "components": ["notifications", "payment"],
    },
    {
        "title": "Search index not updated when records are deleted",
        "body": (
            "When a record is deleted from the database, it still appears in search results.\n"
            "Clicking the result then gives a 404.\n"
            "The search index is only rebuilt on create/update, not on delete.\n\n"
            "Workaround: manually trigger a full reindex (takes ~20 mins).\n"
            "This causes bad UX but no data loss."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P1",
        "has_stack_trace": False,
        "components": ["search", "database"],
    },

    # -----------------------------------------------------------------------
    # BUGS — P2 (Medium)
    # -----------------------------------------------------------------------
    {
        "title": "Search returns no results when query contains an apostrophe",
        "body": (
            "Example: searching \"user's guide\" returns 0 results.\n"
            "Searching \"users guide\" (without apostrophe) returns correct results.\n\n"
            "Looks like the apostrophe is being escaped incorrectly, breaking the query.\n"
            "Possible SQL injection protection interfering with valid input.\n\n"
            "Workaround: users can remove apostrophes from their search."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P2",
        "has_stack_trace": False,
        "components": ["search", "database"],
    },
    {
        "title": "Profile picture upload silently fails for PNG files over 2MB",
        "body": (
            "Steps:\n"
            "1. Go to Profile Settings\n"
            "2. Click 'Change Photo'\n"
            "3. Select a PNG file larger than 2MB\n"
            "4. Click Upload\n\n"
            "Expected: Error message or successful upload\n"
            "Actual: Spinner disappears, no confirmation, photo unchanged\n\n"
            "JPG files of the same size upload fine. No client-side error shown.\n"
            "Server returns 413 Payload Too Large but frontend ignores it."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P2",
        "has_stack_trace": False,
        "components": ["profile"],
    },
    {
        "title": "Notification badge count doesn't reset after reading all notifications",
        "body": (
            "After reading all notifications, the red badge showing the unread count\n"
            "still displays the old number until the page is manually refreshed.\n\n"
            "The notifications are marked read in the database (confirmed via API),\n"
            "but the UI state is not updated. Purely a frontend issue.\n\n"
            "Workaround: refresh the page."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P2",
        "has_stack_trace": False,
        "components": ["notifications", "dashboard"],
    },

    # -----------------------------------------------------------------------
    # BUGS — P3 (Low)
    # -----------------------------------------------------------------------
    {
        "title": "Tooltip text on the export button is misspelled",
        "body": (
            "The tooltip on the Export button on the Reports page says 'Exprot data'\n"
            "instead of 'Export data'. Minor typo, no functional impact."
        ),
        "correct_label": "label_bug",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": ["dashboard"],
    },

    # -----------------------------------------------------------------------
    # FEATURE REQUESTS
    # -----------------------------------------------------------------------
    {
        "title": "Add dark mode to the dashboard",
        "body": (
            "Would love a dark mode option. My eyes hurt after long evening sessions.\n"
            "Many modern apps support this (Notion, Linear, GitHub all have it).\n"
            "Could follow the OS preference automatically via prefers-color-scheme.\n\n"
            "Not urgent, but would improve daily usability a lot."
        ),
        "correct_label": "label_feature",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": ["dashboard"],
    },
    {
        "title": "Allow bulk export of all user records to CSV",
        "body": (
            "Currently I have to export users one at a time. I need to export 500+\n"
            "users for our quarterly board report.\n\n"
            "Request: Add a 'Export All' button to the Users page that downloads\n"
            "a CSV with all fields. Should support filters (date range, role, status).\n\n"
            "This is needed for compliance reporting by end of quarter."
        ),
        "correct_label": "label_feature",
        "correct_severity": "P2",
        "has_stack_trace": False,
        "components": ["dashboard", "api"],
    },
    {
        "title": "Add two-factor authentication support",
        "body": (
            "Our security team requires 2FA for all enterprise accounts.\n"
            "We need this to achieve SOC2 Type II compliance by Q3.\n\n"
            "Should support:\n"
            "- TOTP apps (Google Authenticator, Authy)\n"
            "- SMS backup codes\n"
            "- Recovery codes\n\n"
            "This is a hard requirement for renewing our enterprise contract."
        ),
        "correct_label": "label_feature",
        "correct_severity": "P1",
        "has_stack_trace": False,
        "components": ["auth", "profile"],
    },
    {
        "title": "Support webhook notifications for payment events",
        "body": (
            "We need real-time webhooks for payment events so our internal systems\n"
            "can react immediately instead of polling your API every minute.\n\n"
            "Events we need:\n"
            "  payment.success, payment.failed, payment.refunded\n\n"
            "Please include: event type, payload, and a signature header for verification.\n"
            "Standard stuff — similar to Stripe's webhook format would be ideal."
        ),
        "correct_label": "label_feature",
        "correct_severity": "P2",
        "has_stack_trace": False,
        "components": ["payment", "notifications", "api"],
    },
    {
        "title": "Add keyboard shortcuts for common actions",
        "body": (
            "Power users would benefit from keyboard shortcuts.\n"
            "Suggested:\n"
            "  N — new issue\n"
            "  / — focus search bar\n"
            "  Esc — close modal\n"
            "  Ctrl+S — save\n\n"
            "Low priority but would meaningfully improve workflow speed."
        ),
        "correct_label": "label_feature",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": ["dashboard"],
    },

    # -----------------------------------------------------------------------
    # QUESTIONS
    # -----------------------------------------------------------------------
    {
        "title": "How do I reset my API key?",
        "body": (
            "I accidentally pushed my API key to a public GitHub repo.\n"
            "I've already removed it from the repo but the key may be compromised.\n\n"
            "How do I generate a new API key and revoke the old one?\n"
            "I checked the documentation under 'Settings' but couldn't find the option."
        ),
        "correct_label": "label_question",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": ["api"],
    },
    {
        "title": "What is the rate limit for the Search API endpoint?",
        "body": (
            "I'm building an integration that will call the /api/search endpoint.\n"
            "I need to know the rate limits so I can implement proper backoff.\n\n"
            "The docs mention that rate limits exist but don't specify the numbers.\n"
            "Is it per minute, per hour? What's the 429 retry-after header format?"
        ),
        "correct_label": "label_question",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": ["search", "api"],
    },
    {
        "title": "Can I use the same account across multiple team workspaces?",
        "body": (
            "Our company has two separate workspaces (one for EU, one for US).\n"
            "Can a single user account be a member of both?\n\n"
            "Or do users need to create separate accounts for each workspace?"
        ),
        "correct_label": "label_question",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": ["profile", "auth"],
    },

    # -----------------------------------------------------------------------
    # INVALID
    # -----------------------------------------------------------------------
    {
        "title": "This app is terrible",
        "body": "Everything is broken. Fix it. This is the worst software I've ever used.",
        "correct_label": "label_invalid",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": [],
    },
    {
        "title": "test123",
        "body": "ignore this",
        "correct_label": "label_invalid",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": [],
    },
    {
        "title": "asdfghjkl",
        "body": "zxcvbnm",
        "correct_label": "label_invalid",
        "correct_severity": "P3",
        "has_stack_trace": False,
        "components": [],
    },
]


# ---------------------------------------------------------------------------
# Index templates by ID for deterministic access
# ---------------------------------------------------------------------------
# We assign each template a stable index so that duplicate issues can
# reference the original by index without relying on shuffled positions.

TEMPLATE_INDEX = {i: t for i, t in enumerate(ISSUE_TEMPLATES)}


# ---------------------------------------------------------------------------
# Core generator functions
# ---------------------------------------------------------------------------

def generate_issue(
    template: Optional[Dict] = None,
    issue_id: Optional[str] = None,
    seed_offset: int = 0,
) -> Dict:
    """
    Generate a single issue dict from a template.

    If no template is provided, picks one at random.
    The returned dict contains all fields for BugObservation
    PLUS the hidden ground truth fields (correct_label, correct_severity)
    which are only read by the graders — never shown to the agent.
    """
    if template is None:
        template = random.choice(ISSUE_TEMPLATES)

    return {
        # Observation fields (shown to agent)
        "issue_id": issue_id or f"ISS-{random.randint(1000, 9999)}",
        "title": template["title"],
        "body": template["body"],
        "reporter": REPORTERS[(seed_offset) % len(REPORTERS)],
        "created_at": (
            f"2025-{((seed_offset % 12) + 1):02d}-{((seed_offset * 3 % 28) + 1):02d}"
        ),
        "comments_count": (seed_offset * 7) % 13,
        "has_stack_trace": template["has_stack_trace"],
        "mentioned_components": template["components"],

        # Ground truth fields (hidden from agent, used only by graders)
        "_correct_label": template["correct_label"],
        "_correct_severity": template["correct_severity"],
        "_duplicate_of": template.get("_duplicate_of"),
    }


def generate_inbox(task_difficulty: str) -> List[Dict]:
    """
    Generate the full issue inbox for a given task difficulty.

    Returns a list of issue dicts in the order the agent will see them.
    The _correct_* fields are stripped before sending to the agent
    (this stripping happens in environment.py, not here).

    Inboxes are deterministic given the same difficulty string —
    they do NOT depend on random.seed() so they are reproducible
    across episodes. Duplicates are inserted at fixed positions.
    """

    if task_difficulty == "easy":
        # 8 issues — one of each main type, clearly written, no duplicates.
        # Ordered so the agent sees variety early.
        selected = [
            ISSUE_TEMPLATES[0],   # clear P0 bug (stack trace)
            ISSUE_TEMPLATES[10],  # clear feature
            ISSUE_TEMPLATES[15],  # clear question
            ISSUE_TEMPLATES[3],   # clear P1 bug
            ISSUE_TEMPLATES[11],  # clear feature
            ISSUE_TEMPLATES[18],  # clear invalid
            ISSUE_TEMPLATES[6],   # clear P2 bug
            ISSUE_TEMPLATES[16],  # clear question
        ]
        return [
            generate_issue(t, f"ISS-{1000 + i}", seed_offset=i)
            for i, t in enumerate(selected)
        ]

    elif task_difficulty == "medium":
        # 13 base issues + 2 duplicate pairs = 15 total.
        # Duplicates are paraphrased rewrites of ISS-2000 and ISS-2006.
        base_templates = [
            ISSUE_TEMPLATES[0],   # ISS-2000 — P0 bug (original)
            ISSUE_TEMPLATES[1],   # ISS-2001
            ISSUE_TEMPLATES[3],   # ISS-2002
            ISSUE_TEMPLATES[4],   # ISS-2003
            ISSUE_TEMPLATES[6],   # ISS-2004
            ISSUE_TEMPLATES[7],   # ISS-2005
            ISSUE_TEMPLATES[10],  # ISS-2006 — feature (original)
            ISSUE_TEMPLATES[11],  # ISS-2007
            ISSUE_TEMPLATES[12],  # ISS-2008
            ISSUE_TEMPLATES[13],  # ISS-2009
            ISSUE_TEMPLATES[15],  # ISS-2010
            ISSUE_TEMPLATES[16],  # ISS-2011
            ISSUE_TEMPLATES[18],  # ISS-2012
        ]
        issues = [
            generate_issue(t, f"ISS-{2000 + i}", seed_offset=i)
            for i, t in enumerate(base_templates)
        ]

        # Duplicate 1: paraphrased version of ISS-2000 (login fails on Safari)
        dup1 = generate_issue(ISSUE_TEMPLATES[0], "ISS-2013", seed_offset=13)
        dup1["title"] = "Can't log in on iPhone — server error every time"
        dup1["body"] = (
            "Tried logging in from my iPhone and keep getting a server error.\n"
            "Works fine from my laptop. Didn't change my password or anything.\n"
            "Started happening a couple of days ago."
        )
        dup1["has_stack_trace"] = False
        dup1["_correct_label"] = "label_duplicate"
        dup1["_duplicate_of"] = "ISS-2000"

        # Duplicate 2: paraphrased version of ISS-2006 (dark mode feature)
        dup2 = generate_issue(ISSUE_TEMPLATES[10], "ISS-2014", seed_offset=14)
        dup2["title"] = "Please add a night mode to reduce eye strain"
        dup2["body"] = (
            "I work late at night and the bright white interface is really hard\n"
            "on my eyes. A dark or night mode option would be very welcome.\n"
            "Something like what GitHub or VS Code has would be perfect."
        )
        dup2["_correct_label"] = "label_duplicate"
        dup2["_duplicate_of"] = "ISS-2006"

        issues.extend([dup1, dup2])

        # Shuffle middle section only (keep first issue as obvious anchor)
        tail = issues[1:]
        random.Random(42).shuffle(tail)
        return [issues[0]] + tail

    else:  # hard
        # 18 base issues + 2 subtle duplicate pairs = 20 total.
        # Subtle duplicates use completely different wording.
        base_templates = [
            ISSUE_TEMPLATES[0],   # ISS-3000 — P0 (original)
            ISSUE_TEMPLATES[1],   # ISS-3001 — P0 (original)
            ISSUE_TEMPLATES[2],   # ISS-3002
            ISSUE_TEMPLATES[3],   # ISS-3003
            ISSUE_TEMPLATES[4],   # ISS-3004
            ISSUE_TEMPLATES[5],   # ISS-3005
            ISSUE_TEMPLATES[6],   # ISS-3006
            ISSUE_TEMPLATES[7],   # ISS-3007
            ISSUE_TEMPLATES[8],   # ISS-3008
            ISSUE_TEMPLATES[9],   # ISS-3009
            ISSUE_TEMPLATES[10],  # ISS-3010 — feature (original)
            ISSUE_TEMPLATES[11],  # ISS-3011
            ISSUE_TEMPLATES[12],  # ISS-3012
            ISSUE_TEMPLATES[13],  # ISS-3013
            ISSUE_TEMPLATES[14],  # ISS-3014
            ISSUE_TEMPLATES[15],  # ISS-3015
            ISSUE_TEMPLATES[16],  # ISS-3016
            ISSUE_TEMPLATES[18],  # ISS-3017
        ]
        issues = [
            generate_issue(t, f"ISS-{3000 + i}", seed_offset=i)
            for i, t in enumerate(base_templates)
        ]

        # Subtle duplicate 1: different wording for ISS-3001 (payment hang >$999)
        subtle_dup1 = generate_issue(ISSUE_TEMPLATES[1], "ISS-3018", seed_offset=18)
        subtle_dup1["title"] = "Large payments never complete"
        subtle_dup1["body"] = (
            "Noticed that some payments just never go through.\n"
            "No error is shown — it just keeps loading.\n"
            "Smaller amounts seem to work. Larger ones don't.\n"
            "Not sure if this is on our end or yours."
        )
        subtle_dup1["has_stack_trace"] = False
        subtle_dup1["_correct_label"] = "label_duplicate"
        subtle_dup1["_duplicate_of"] = "ISS-3001"

        # Subtle duplicate 2: different wording for ISS-3004 (duplicate emails)
        subtle_dup2 = generate_issue(ISSUE_TEMPLATES[4], "ISS-3019", seed_offset=19)
        subtle_dup2["title"] = "I'm receiving two emails when I pay"
        subtle_dup2["body"] = (
            "Every time I make a payment I get two identical confirmation emails\n"
            "within seconds of each other. Slightly annoying. Not a blocker\n"
            "but probably something worth looking at."
        )
        subtle_dup2["_correct_label"] = "label_duplicate"
        subtle_dup2["_duplicate_of"] = "ISS-3004"

        issues.extend([subtle_dup1, subtle_dup2])

        # Full shuffle — no anchoring, agent must rely purely on content
        random.Random(99).shuffle(issues)
        return issues


# ---------------------------------------------------------------------------
# Utility: strip hidden fields before sending to agent
# ---------------------------------------------------------------------------

def strip_ground_truth(issue: Dict) -> Dict:
    """
    Remove the _correct_* hidden fields from an issue dict.
    Call this before constructing a BugObservation.
    """
    return {k: v for k, v in issue.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python issue_generator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Easy inbox (8 issues) ===")
    easy = generate_inbox("easy")
    for issue in easy:
        label = issue["_correct_label"]
        sev = issue["_correct_severity"]
        print(f"  [{issue['issue_id']}] [{label}] [{sev}] {issue['title'][:60]}")

    print(f"\n=== Medium inbox ({len(generate_inbox('medium'))} issues) ===")
    for issue in generate_inbox("medium"):
        label = issue["_correct_label"]
        dup = f" → dup of {issue['_duplicate_of']}" if issue.get("_duplicate_of") else ""
        print(f"  [{issue['issue_id']}] [{label}]{dup} {issue['title'][:55]}")

    print(f"\n=== Hard inbox ({len(generate_inbox('hard'))} issues) ===")
    for issue in generate_inbox("hard"):
        label = issue["_correct_label"]
        dup = f" → dup of {issue['_duplicate_of']}" if issue.get("_duplicate_of") else ""
        print(f"  [{issue['issue_id']}] [{label}]{dup} {issue['title'][:55]}")
