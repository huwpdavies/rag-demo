"""Generate samples/fernhollow-handbook.pdf, a fictional company document for demos.

Fernhollow Outdoor Gear Ltd does not exist. The handbook is written for a RAG
demonstration: specific, checkable policies that a language model cannot know
from general knowledge, some answers that need facts from several sections,
and some deliberate gaps (e.g. pensions, dress code) so the "the document does
not say" behaviour can be shown.

This script is not part of the rag-demo package. To regenerate the PDF after
editing the text below:

    pip install reportlab
    python samples/make_handbook.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT = Path(__file__).with_name("fernhollow-handbook.pdf")
COMPANY = "Fernhollow Outdoor Gear Ltd"
FOOTER = f"{COMPANY} is a fictional company. This document was created for a RAG demonstration."

# Bullets are drawn with an embedded TrueType font: the built-in Helvetica maps "•" to a character
# that text extraction reads back as a control code. Falls back to a dash where Arial is missing.
ARIAL = Path("C:/Windows/Fonts/arial.ttf")
if ARIAL.is_file():
    pdfmetrics.registerFont(TTFont("Arial", str(ARIAL)))
    BULLET_FONT, BULLET_CHAR = "Arial", "•"
else:
    BULLET_FONT, BULLET_CHAR = "Helvetica", "-"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("title", parent=styles["Title"], fontSize=20, spaceAfter=2 * mm, textColor=colors.HexColor("#1f4d3a"))
SUBTITLE = ParagraphStyle("subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9.5, textColor=colors.HexColor("#555555"), spaceAfter=5 * mm)
H2 = ParagraphStyle(
    "h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=4 * mm, spaceAfter=1.5 * mm,
    textColor=colors.HexColor("#1f4d3a"), keepWithNext=1,  # never leave a heading alone at the foot of a page
)
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=13.2, spaceAfter=1.8 * mm)
BULLET = ParagraphStyle("bullet", parent=BODY, spaceAfter=0.8 * mm)


def h(text):
    return Paragraph(text, H2)


def p(text):
    return Paragraph(text, BODY)


def bullets(*items):
    return ListFlowable(
        [ListItem(Paragraph(item, BULLET), leftIndent=4 * mm) for item in items],
        bulletType="bullet", start=BULLET_CHAR, leftIndent=4 * mm, bulletFontSize=9, bulletFontName=BULLET_FONT,
    )


story = [
    Paragraph("Fernhollow Employee Handbook", TITLE),
    Paragraph("Policy summary · Version 4.2 · Effective 1 January 2026 · Owner: People Team", SUBTITLE),

    h("1. Welcome to Fernhollow"),
    p(f"{COMPANY} designs and sells tents, packs and waterproof clothing from our head office and "
      "workshop in Kendal, Cumbria. The company was founded in 2011 by climbers Maya Okafor and Tom Reddish "
      "and now employs around 180 people across the office, the workshop and our Penrith warehouse. "
      "This handbook summarises the policies you are most likely to need. Where a policy here and your "
      "employment contract disagree, your contract takes precedence."),

    h("2. Working hours and hybrid working"),
    p("The standard full-time week is 37.5 hours. Everyone is expected to be available during core hours, "
      "10:00 to 15:00 Monday to Thursday and 10:00 to 13:00 on Fridays (our “Early Friday”). Outside core "
      "hours you can arrange your time with your manager."),
    bullets(
        "Office-based roles are hybrid: work from the Kendal office at least <b>two days a week</b>. "
        "Tuesday is Team Day, when everyone in an office-based role is expected in.",
        "Workshop and warehouse roles are fully on site.",
        "You may work from abroad for up to <b>20 working days per calendar year</b>, with your manager’s "
        "approval and at least four weeks’ notice to the People Team, who check tax and insurance.",
        "New starters in hybrid roles receive a one-off <b>home office allowance of £300</b>, plus £150 every "
        "three years towards replacement equipment.",
    ),

    h("3. Annual leave"),
    p("Full-time employees receive <b>27 days of annual leave</b> plus the 8 UK bank holidays; part-time "
      "allowances are pro rata. You also get an extra day off in the week of your birthday."),
    bullets(
        "Long service: one extra day a year after 3 years’ service, and three extra days after 5 years "
        "(so 30 days in total).",
        "You can carry over up to <b>5 unused days</b> into the next year; carried-over days must be used by 31 March.",
        "Book leave in the Ridgeline HR portal. Requests for more than five consecutive working days need "
        "four weeks’ notice.",
        "Warehouse staff cannot take leave between 1 and 31 December, our peak shipping season.",
    ),

    h("4. Sickness absence"),
    p("If you are unwell, tell your manager by phone or message <b>before 9:30</b> on your first day of "
      "absence, and keep them updated. You can self-certify for up to 7 calendar days; after that you need "
      "a fit note from a doctor. After your probation period (3 months), Fernhollow pays full salary for up "
      "to <b>15 working days of sickness</b> in any rolling 12-month period; after that, Statutory Sick Pay "
      "applies. Any absence of more than 3 days is followed by a short return-to-work conversation."),

    h("5. Family leave"),
    p("Family leave is available from your first day. The primary carer of a new child (through birth, "
      "adoption or surrogacy) receives <b>20 weeks at full pay</b>, followed by statutory pay. Partners and "
      "secondary carers receive <b>6 weeks at full pay</b>, which can be taken in blocks at any time during "
      "the child’s first year. Tell the People Team at least 15 weeks before the expected week of birth or placement."),

    h("6. Expenses and travel"),
    p("Claim expenses in the Tally app within <b>30 days</b> of spending the money. Receipts are required for "
      "anything over £10. The lowest reasonable cost applies to all bookings."),
    bullets(
        "<b>Rail:</b> standard class. First class is allowed when the journey is longer than 3 hours each way.",
        "<b>Flights:</b> economy for flights under 6 hours, premium economy for longer flights. Business class "
        "needs the CEO’s approval.",
        "<b>Hotels:</b> up to £140 a night outside London and £200 a night in London, including breakfast.",
        "<b>Meals:</b> up to £35 a day when staying away overnight, or £12 for lunch on a day trip lasting "
        "more than 5 hours.",
        "<b>Mileage</b> in your own car: 45p a mile for the first 10,000 business miles in the tax year, then "
        "25p. Business journeys by bicycle: 20p a mile.",
        "<b>Client entertainment:</b> up to £60 per person. Alcohol is only reimbursed as part of client entertainment.",
    ),
    p("Any single claim over £500 needs approval from your Head of Department; claims over £2,000 also need "
      "the Finance Director."),

    h("7. IT, security and AI tools"),
    bullets(
        "Multi-factor authentication is required on every company account. Passwords must be at least "
        "<b>14 characters</b> and kept in the company password manager.",
        "If you receive a suspected phishing email, do not click anything. Use the Report Phish button in "
        "Outlook, or forward it to security@fernhollow.example, <b>within 1 hour</b>.",
        "Report a lost or stolen laptop or phone to the IT Service Desk <b>within 24 hours</b> on extension "
        "4357 (the line is staffed 24/7), so it can be locked and wiped remotely.",
        "Personal USB storage devices must not be connected to company laptops. Lock your screen whenever "
        "you leave your desk.",
        "Only use AI tools on the approved list on the intranet. Never paste customer personal data or "
        "unreleased product designs into an AI tool that is not on the approved list.",
    ),

    h("8. Learning and development"),
    p("Every employee has a learning budget of <b>£1,000 a year</b> and 3 paid learning days for courses, "
      "conferences or study. Membership fees for relevant professional bodies are reimbursed. If you leave "
      "within 12 months of completing a course that cost more than £2,000, you repay 50% of its cost."),

    h("9. Wellbeing and benefits"),
    bullets(
        "Employee Assistance Programme: a free, confidential helpline available 24 hours a day on 01632 960 123, "
        "offering counselling and legal and financial advice.",
        "Cycle to Work scheme for bikes and equipment up to £2,500, repaid through salary sacrifice.",
        "Staff discount: <b>40% off Fernhollow products</b>, up to £1,500 of retail value a year.",
        "Two paid volunteering days a year for a charity of your choice.",
    ),

    h("10. Gifts and hospitality"),
    p("You may accept small gifts and hospitality from suppliers or customers, such as a meal or a branded "
      "item, as long as they could not be seen to influence a decision. Record anything worth more than "
      "<b>£50</b> in the gifts register on the intranet within 5 working days. Never accept a gift or "
      "hospitality of any value from a supplier who is taking part in a tender we are running, and never "
      "accept cash or vouchers."),

    h("11. Leaving Fernhollow"),
    p("The notice period is <b>one month</b> for employees with less than two years’ service and "
      "<b>two months</b> after that; people managers give three months at any stage. Unused annual leave is "
      "paid in your final salary. Return your laptop, phone, security pass and any sample products to the "
      "People Team within 5 working days of your last day. Everyone who leaves is offered an exit "
      "conversation with someone outside their own team."),

    h("12. Key contacts"),
]

contacts = [
    ["Team", "Contact", "Use for"],
    ["People Team", "people@fernhollow.example", "Leave, family leave, working abroad"],
    ["IT Service Desk", "Extension 4357 (24/7)", "Lost devices, access problems"],
    ["Security", "security@fernhollow.example", "Phishing and security incidents"],
    ["Finance", "expenses@fernhollow.example", "Expense claims and approvals"],
    ["Wellbeing", "01632 960 123 (24/7)", "Employee Assistance Programme"],
]
table = Table(contacts, colWidths=[32 * mm, 58 * mm, 72 * mm])
table.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
    ("FONT", (0, 1), (-1, -1), "Helvetica", 9.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeee6")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fbfae")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story += [table, Spacer(1, 3 * mm)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, 10 * mm, FOOTER)
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
    title="Fernhollow Employee Handbook", author=f"{COMPANY} (fictional)",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUTPUT}")
